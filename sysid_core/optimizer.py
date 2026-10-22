"""Optimization engine for grey-box system identification via BPTT.

Combines Adam and L-BFGS with Huber trajectory loss and physical parameter
regularization to solve inverse dynamics problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from sysid_core.base_model import DifferentiableDynamicalSystem
from sysid_core.rk4_integrator import RK4Integrator


@dataclass
class OptimizationResult:
    """System identification convergence statistics and parameter history."""

    initial_params: Dict[str, float]
    optimized_params: Dict[str, float]
    loss_history: List[float] = field(default_factory=list)
    param_history: Dict[str, List[float]] = field(default_factory=dict)
    initial_rmse: float = 0.0
    final_rmse: float = 0.0
    status: str = "success"


class SystemIdentificationOptimizer:
    """Orchestrates trajectory-based parameter identification using PyTorch autograd."""

    def __init__(
        self,
        model: DifferentiableDynamicalSystem,
        dt: float,
        state_weights: Optional[torch.Tensor] = None,
        huber_delta: float = 1.0,
        reg_weight: float = 0.01,
        loss_type: str = "huber",
    ) -> None:
        self.model: DifferentiableDynamicalSystem = model
        self.dt: float = float(dt)
        self.integrator: RK4Integrator = RK4Integrator(dt=self.dt, hold_mode="linear", n_substeps=1)
        self.huber_delta: float = float(huber_delta)
        self.reg_weight: float = float(reg_weight)
        self.loss_type: str = loss_type

        # Default weights normalize typical order-of-magnitude differences across states
        if state_weights is not None:
            self.state_weights = state_weights
        elif hasattr(model, "observable_dim") and model.observable_dim == 6:
            # Default weights for Ackermann [p_x, p_y, psi, v_x, v_y, omega_z]
            self.state_weights = torch.tensor(
                [1.0, 1.0, 2.0, 1.0, 1.5, 1.0],
                dtype=torch.float64,
            )
        else:
            dim = getattr(model, "observable_dim", getattr(model, "state_dim", 1))
            self.state_weights = torch.ones(dim, dtype=torch.float64)

    def _compute_trajectory_loss(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted trajectory loss (Huber or MSE)."""
        min_len = min(y_pred.shape[-2], y_target.shape[-2])
        diff = y_pred[..., :min_len, :] - y_target[..., :min_len, :]
        weights = self.state_weights.to(dtype=diff.dtype, device=diff.device)
        if weights.shape[-1] != diff.shape[-1]:
            weights = torch.ones(diff.shape[-1], dtype=diff.dtype, device=diff.device)

        if self.loss_type == "huber":
            abs_diff = torch.abs(diff)
            delta = self.huber_delta
            huber_element = torch.where(
                abs_diff <= delta,
                0.5 * abs_diff.pow(2),
                delta * (abs_diff - 0.5 * delta),
            )
            weighted_elem = huber_element * weights
            return torch.mean(torch.sum(weighted_elem, dim=-1))
        else:
            squared_err = diff.pow(2) * weights
            return torch.mean(torch.sum(squared_err, dim=-1))

    def evaluate_rollout(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out full state and observable predictions."""
        x_pred = self.integrator.rollout(
            ode_fn=self.model.forward,
            x0=x0,
            u_seq=u_seq,
            return_initial_state=False,
        )
        y_pred = self.model.observe(x_pred)
        return x_pred, y_pred

    def compute_rmse(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        y_target: torch.Tensor,
    ) -> float:
        """Calculate Root Mean Square Error over observable trajectory."""
        with torch.no_grad():
            _, y_pred = self.evaluate_rollout(x0, u_seq)
            min_len = min(y_pred.shape[-2], y_target.shape[-2])
            diff = y_pred[..., :min_len, :] - y_target[..., :min_len, :]
            mse = torch.mean(diff.pow(2)).item()
            return float(torch.sqrt(torch.tensor(mse)).item())

    def fit(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        y_target: torch.Tensor,
        adam_epochs: int = 120,
        adam_lr: float = 0.05,
        lbfgs_epochs: int = 30,
        lbfgs_lr: float = 0.5,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    ) -> OptimizationResult:
        """Run two-phase optimization (Adam -> L-BFGS) to estimate parameters.

        Args:
            x0: Initial full state vector [state_dim].
            u_seq: Control sequence [num_timesteps, control_dim].
            y_target: Measured ground truth observable trajectory [num_timesteps, obs_dim].
            adam_epochs: Iterations of Adam gradient descent.
            adam_lr: Learning rate for Adam.
            lbfgs_epochs: Iterations of second-order L-BFGS.
            lbfgs_lr: Learning rate for L-BFGS.
            verbose: If True, prints epoch progress.
            callback: Optional hook called after each iteration.

        Returns:
            OptimizationResult containing parameter convergence logs.
        """
        # Detach inputs to avoid unintended backprop through input tensors
        x0 = x0.detach()
        u_seq = u_seq.detach()
        y_target = y_target.detach()

        init_params = self.model.get_physical_parameters()
        init_rmse = self.compute_rmse(x0, u_seq, y_target)

        loss_history: List[float] = []
        param_history: Dict[str, List[float]] = {k: [v] for k, v in init_params.items()}

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Model has no trainable parameters registered with requires_grad=True")

        iteration_counter = 0

        # Phase 1: Adam for robust exploration
        if adam_epochs > 0:
            adam_opt = torch.optim.Adam(trainable_params, lr=adam_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                adam_opt, T_max=adam_epochs, eta_min=adam_lr * 0.1
            )

            for epoch in range(adam_epochs):
                adam_opt.zero_grad()
                _, y_pred = self.evaluate_rollout(x0, u_seq)
                traj_loss = self._compute_trajectory_loss(y_pred, y_target)
                reg_loss = self.model.prior_regularization_loss()
                total_loss = traj_loss + self.reg_weight * reg_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
                adam_opt.step()
                scheduler.step()

                loss_val = total_loss.item()
                loss_history.append(loss_val)
                iteration_counter += 1

                curr_params = self.model.get_physical_parameters()
                for k, v in curr_params.items():
                    param_history[k].append(v)

                if callback is not None:
                    callback(iteration_counter, loss_val, curr_params)

                if verbose and (epoch % max(1, adam_epochs // 5) == 0 or epoch == adam_epochs - 1):
                    print(
                        f"[Adam Epoch {epoch:3d}/{adam_epochs}] "
                        f"Loss: {loss_val:.6f} (Traj: {traj_loss.item():.6f}, Reg: {reg_loss.item():.6f})"
                    )

        # Phase 2: L-BFGS for fine convergence
        if lbfgs_epochs > 0:
            lbfgs_opt = torch.optim.LBFGS(
                trainable_params,
                lr=lbfgs_lr,
                max_iter=20,
                history_size=15,
                line_search_fn="strong_wolfe",
            )

            for epoch in range(lbfgs_epochs):
                current_epoch_loss = [0.0]

                def closure() -> torch.Tensor:
                    lbfgs_opt.zero_grad()
                    _, y_pred_inner = self.evaluate_rollout(x0, u_seq)
                    traj_loss_inner = self._compute_trajectory_loss(y_pred_inner, y_target)
                    reg_loss_inner = self.model.prior_regularization_loss()
                    loss_inner = traj_loss_inner + self.reg_weight * reg_loss_inner
                    if not torch.isfinite(loss_inner):
                        return torch.tensor(1e6, dtype=loss_inner.dtype, device=loss_inner.device, requires_grad=True)
                    loss_inner.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
                    current_epoch_loss[0] = loss_inner.item()
                    return loss_inner

                lbfgs_opt.step(closure)
                loss_val = current_epoch_loss[0]
                loss_history.append(loss_val)
                iteration_counter += 1

                curr_params = self.model.get_physical_parameters()
                for k, v in curr_params.items():
                    param_history[k].append(v)

                if callback is not None:
                    callback(iteration_counter, loss_val, curr_params)

                if verbose and (epoch % max(1, lbfgs_epochs // 3) == 0 or epoch == lbfgs_epochs - 1):
                    print(f"[L-BFGS Epoch {epoch:2d}/{lbfgs_epochs}] Loss: {loss_val:.6f}")

        final_rmse = self.compute_rmse(x0, u_seq, y_target)
        final_params = self.model.get_physical_parameters()

        return OptimizationResult(
            initial_params=init_params,
            optimized_params=final_params,
            loss_history=loss_history,
            param_history=param_history,
            initial_rmse=init_rmse,
            final_rmse=final_rmse,
            status="converged",
        )
