"""Differentiable 4th-Order Runge-Kutta (RK4) ODE Integrator in pure PyTorch.

Allows gradients to backpropagate through time (BPTT) across multi-step trajectory
rollouts for grey-box system identification.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple, Union
import torch


def rk4_step(
    ode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    u_next: Optional[torch.Tensor] = None,
    hold_mode: Literal["zoh", "linear"] = "linear",
) -> torch.Tensor:
    """Perform a single 4th-order Runge-Kutta integration step.

    Solves dx/dt = f(t, x, u) across the interval [t, t + dt].

    Args:
        ode_fn: Continuous-time system derivative function f(t, x, u).
        t: Current time coordinate [s].
        x: Current state vector of shape (..., state_dim).
        u: Control vector at start of interval of shape (..., control_dim).
        dt: Time step duration [s].
        u_next: Optional control vector at end of interval of shape (..., control_dim).
        hold_mode: Interpolation scheme between u and u_next ('zoh' or 'linear').

    Returns:
        Next state vector x(t + dt) of shape (..., state_dim).
    """
    dt_tensor = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
    half_dt = 0.5 * dt_tensor

    if hold_mode == "linear" and u_next is not None:
        u_mid = 0.5 * (u + u_next)
        u_end = u_next
    else:
        u_mid = u
        u_end = u

    t_mid = t + half_dt
    t_end = t + dt_tensor

    k1 = ode_fn(t, x, u)
    k2 = ode_fn(t_mid, x + half_dt * k1, u_mid)
    k3 = ode_fn(t_mid, x + half_dt * k2, u_mid)
    k4 = ode_fn(t_end, x + dt_tensor * k3, u_end)

    return x + (dt_tensor / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_rollout(
    ode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    t0: float = 0.0,
    hold_mode: Literal["zoh", "linear"] = "linear",
    n_substeps: int = 1,
    return_initial_state: bool = False,
) -> torch.Tensor:
    """Simulate a multi-step trajectory rollout using RK4 integration.

    Args:
        ode_fn: Continuous-time ODE f(t, x, u).
        x0: Initial state vector of shape (..., state_dim).
        u_seq: Control sequence of shape (..., num_steps, control_dim).
        dt: Sampling time duration per control step [s].
        t0: Initial simulation time [s].
        hold_mode: 'zoh' (zero-order hold) or 'linear' (first-order hold).
        n_substeps: Number of internal RK4 sub-steps per control step dt.
        return_initial_state: If True, returns trajectory of length num_steps + 1
                              starting with x0. If False, returns trajectory of
                              length num_steps matching u_seq timesteps.

    Returns:
        Trajectory tensor of states rolled out over time.
    """
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")

    # Normalize 1D control sequence (T,) -> (T, 1)
    if u_seq.ndim == 1:
        u_seq = u_seq.unsqueeze(-1)

    # Determine batch dimensions and sequence length
    is_batched = (u_seq.ndim >= 3) or (x0.ndim >= 2)
    num_steps = u_seq.shape[-2]

    sub_dt = dt / float(n_substeps)
    curr_x = x0
    curr_t = torch.tensor(t0, dtype=x0.dtype, device=x0.device)

    trajectory = []
    if return_initial_state:
        trajectory.append(curr_x)

    for step_idx in range(num_steps):
        if u_seq.ndim >= 3:
            u_k = u_seq[..., step_idx, :]
            if step_idx + 1 < num_steps:
                u_next = u_seq[..., step_idx + 1, :]
            else:
                u_next = u_k
        else:
            u_k = u_seq[step_idx, :]
            if step_idx + 1 < num_steps:
                u_next = u_seq[step_idx + 1, :]
            else:
                u_next = u_k

        # Integrate through internal sub-steps
        for sub in range(n_substeps):
            if hold_mode == "linear":
                alpha_start = sub / float(n_substeps)
                alpha_end = (sub + 1.0) / float(n_substeps)
                sub_u = (1.0 - alpha_start) * u_k + alpha_start * u_next
                sub_u_next = (1.0 - alpha_end) * u_k + alpha_end * u_next
            else:
                sub_u = u_k
                sub_u_next = u_k

            curr_x = rk4_step(
                ode_fn=ode_fn,
                t=curr_t,
                x=curr_x,
                u=sub_u,
                dt=sub_dt,
                u_next=sub_u_next,
                hold_mode=hold_mode,
            )
            curr_t = curr_t + sub_dt

        trajectory.append(curr_x)

    # Stack along time dimension: shape (..., T, state_dim)
    dim_to_stack = -2 if is_batched else 0
    return torch.stack(trajectory, dim=dim_to_stack)


class RK4Integrator:
    """Configurable RK4 Integrator wrapper object for training and simulation."""

    def __init__(
        self,
        dt: float,
        hold_mode: Literal["zoh", "linear"] = "linear",
        n_substeps: int = 1,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"Integration dt must be positive, got {dt}")
        self.dt: float = float(dt)
        self.hold_mode: Literal["zoh", "linear"] = hold_mode
        self.n_substeps: int = int(n_substeps)

    def step(
        self,
        ode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        u_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single integration step."""
        return rk4_step(
            ode_fn=ode_fn,
            t=t,
            x=x,
            u=u,
            dt=self.dt,
            u_next=u_next,
            hold_mode=self.hold_mode,
        )

    def rollout(
        self,
        ode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        t0: float = 0.0,
        return_initial_state: bool = False,
    ) -> torch.Tensor:
        """Roll out trajectory given control sequence."""
        return rk4_rollout(
            ode_fn=ode_fn,
            x0=x0,
            u_seq=u_seq,
            dt=self.dt,
            t0=t0,
            hold_mode=self.hold_mode,
            n_substeps=self.n_substeps,
            return_initial_state=return_initial_state,
        )
