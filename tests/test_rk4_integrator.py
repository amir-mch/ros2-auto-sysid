"""Unit tests for pure PyTorch differentiable RK4 ODE integrator."""

import math
import pytest
import torch

from sysid_core.rk4_integrator import RK4Integrator, rk4_step, rk4_rollout


def test_rk4_step_linear_decay():
    """Verify RK4 accuracy against analytical solution of exponential decay."""
    decay_rate = 2.5

    def linear_decay_ode(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -decay_rate * x

    x0 = torch.tensor([4.0], dtype=torch.float64)
    dt = 0.05
    t0 = torch.tensor(0.0, dtype=torch.float64)
    u0 = torch.tensor([0.0], dtype=torch.float64)

    x1 = rk4_step(linear_decay_ode, t0, x0, u0, dt=dt)
    expected_x1 = x0 * math.exp(-decay_rate * dt)

    assert torch.allclose(x1, expected_x1, atol=1e-6)


def test_rk4_convergence_rate():
    """Verify 4th-order global convergence rate: error decreases by ~16x when dt is halved."""
    decay_rate = 1.5

    def ode_fn(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -decay_rate * x

    t_end = 1.0
    x0 = torch.tensor([1.0], dtype=torch.float64)
    expected_final = x0 * math.exp(-decay_rate * t_end)

    # Coarse step
    dt_coarse = 0.1
    n_coarse = int(t_end / dt_coarse)
    u_seq_coarse = torch.zeros(n_coarse, 1, dtype=torch.float64)
    traj_coarse = rk4_rollout(ode_fn, x0, u_seq_coarse, dt=dt_coarse)
    err_coarse = torch.abs(traj_coarse[-1] - expected_final).item()

    # Fine step (halved)
    dt_fine = 0.05
    n_fine = int(t_end / dt_fine)
    u_seq_fine = torch.zeros(n_fine, 1, dtype=torch.float64)
    traj_fine = rk4_rollout(ode_fn, x0, u_seq_fine, dt=dt_fine)
    err_fine = torch.abs(traj_fine[-1] - expected_final).item()

    # For 4th order, error ratio should be close to 2^4 = 16
    ratio = err_coarse / err_fine
    assert 14.0 <= ratio <= 18.0, f"Expected 4th order ratio ~16, got {ratio:.2f}"


def test_rk4_autograd_gradients():
    """Verify gradients propagate backwards through time across multiple integration steps."""
    param = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)

    def parameterized_ode(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -param * x + u

    x0 = torch.tensor([2.0], dtype=torch.float64)
    u_seq = torch.ones(20, 1, dtype=torch.float64)
    dt = 0.05

    traj = rk4_rollout(parameterized_ode, x0, u_seq, dt=dt)
    loss = torch.sum(traj.pow(2))
    loss.backward()

    assert param.grad is not None
    assert not torch.isnan(param.grad)
    assert param.grad.abs().item() > 0.0


def test_rk4_batching_equivalence():
    """Verify batched rollout produces identical output to individual rollouts."""
    def oscillator(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # State: [pos, vel]
        p = x[..., 0]
        v = x[..., 1]
        dp = v
        dv = -4.0 * p + u[..., 0]
        return torch.stack([dp, dv], dim=-1)

    batch_x0 = torch.tensor([[1.0, 0.0], [0.5, 0.2]], dtype=torch.float64)
    batch_u = torch.randn(2, 30, 1, dtype=torch.float64)
    dt = 0.02

    batched_traj = rk4_rollout(oscillator, batch_x0, batch_u, dt=dt)

    single_traj_0 = rk4_rollout(oscillator, batch_x0[0], batch_u[0], dt=dt)
    single_traj_1 = rk4_rollout(oscillator, batch_x0[1], batch_u[1], dt=dt)

    assert torch.allclose(batched_traj[0], single_traj_0, atol=1e-10)
    assert torch.allclose(batched_traj[1], single_traj_1, atol=1e-10)


def test_rk4_batched_x0_unbatched_u():
    """Verify batched initial states with shared unbatched control sequence produce (B, T, D)."""
    def decay(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -x + u

    x0 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)  # shape (3, 1)
    u_seq = torch.ones(15, 1, dtype=torch.float64)  # shape (15, 1)
    traj = rk4_rollout(decay, x0, u_seq, dt=0.05)

    assert traj.shape == torch.Size([3, 15, 1])


def test_rk4_1d_u_seq():
    """Verify 1D control sequence (T,) is supported without crashing."""
    def simple_sys(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -0.5 * x + u[..., 0]

    x0 = torch.tensor([1.0], dtype=torch.float64)
    u_seq_1d = torch.ones(20, dtype=torch.float64)  # 1D tensor
    traj = rk4_rollout(simple_sys, x0, u_seq_1d, dt=0.05)

    assert traj.shape == torch.Size([20, 1])
    assert not torch.isnan(traj).any()


def test_rk4_return_initial_state():
    """Verify return_initial_state=True prepends x0 to output trajectory."""
    def decay(t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -x

    x0 = torch.tensor([5.0], dtype=torch.float64)
    u_seq = torch.zeros(10, 1, dtype=torch.float64)
    traj = rk4_rollout(decay, x0, u_seq, dt=0.1, return_initial_state=True)

    assert traj.shape == torch.Size([11, 1])
    assert torch.allclose(traj[0], x0)
