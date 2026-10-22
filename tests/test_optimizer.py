"""Unit tests for SystemIdentificationOptimizer."""

import torch
import pytest

from sysid_core.base_model import DifferentiableDynamicalSystem
from sysid_core.parameter import PhysicalParameter
from sysid_core.rk4_integrator import RK4Integrator
from sysid_core.optimizer import SystemIdentificationOptimizer


class SimpleFirstOrderSystem(DifferentiableDynamicalSystem):
    """Linear system dx/dt = -a * x + b * u for optimizer unit testing."""

    def __init__(self, a: float = 2.0, b: float = 1.0) -> None:
        super().__init__(state_dim=1, control_dim=1)
        self.a = self.register_physical_parameter(
            PhysicalParameter("a", a, (0.1, 10.0), unit="1/s")
        )
        self.b = self.register_physical_parameter(
            PhysicalParameter("b", b, (0.1, 10.0), unit="m/(s^2*V)")
        )

    @property
    def state_names(self):
        return ["x"]

    @property
    def control_names(self):
        return ["u"]

    def ode(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -self.a() * x + self.b() * u


def test_optimizer_recovers_parameters():
    """Verify optimizer recovers parameters on a noiseless first order system."""
    torch.manual_seed(123)
    dt = 0.05
    integrator = RK4Integrator(dt=dt)

    # True system with a=3.0, b=2.0
    true_sys = SimpleFirstOrderSystem(a=3.0, b=2.0)
    x0 = torch.tensor([1.0], dtype=torch.float64)

    # Square wave excitation sequence
    u_seq = torch.sin(torch.linspace(0, 10, 80, dtype=torch.float64)).unsqueeze(-1)
    with torch.no_grad():
        y_target = integrator.rollout(true_sys.forward, x0, u_seq)

    # Estimated system with perturbed initial guess: a=1.0, b=3.5
    est_sys = SimpleFirstOrderSystem(a=1.0, b=3.5)
    optimizer = SystemIdentificationOptimizer(est_sys, dt=dt, huber_delta=0.5, reg_weight=0.0)

    initial_rmse = optimizer.compute_rmse(x0, u_seq, y_target)
    result = optimizer.fit(
        x0=x0,
        u_seq=u_seq,
        y_target=y_target,
        adam_epochs=35,
        adam_lr=0.1,
        lbfgs_epochs=10,
        lbfgs_lr=0.5,
        verbose=False,
    )

    final_rmse = result.final_rmse
    assert final_rmse < initial_rmse * 0.1, f"Expected RMSE reduction, got {initial_rmse} -> {final_rmse}"

    params = est_sys.get_physical_parameters()
    assert abs(params["a"] - 3.0) < 0.2
    assert abs(params["b"] - 2.0) < 0.2


def test_optimizer_differential_drive_5state():
    """Verify optimizer adapts to 5-state DifferentialDriveModel without weight mismatch crash."""
    from sysid_models.differential_drive import DifferentialDriveModel

    dt = 0.05
    integrator = RK4Integrator(dt=dt)
    true_robot = DifferentialDriveModel(mass=6.0, tau_v=0.12)
    x0 = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0], dtype=torch.float64)
    u_seq = torch.tensor([[1.5, 0.2]] * 40, dtype=torch.float64)

    with torch.no_grad():
        y_target = integrator.rollout(true_robot.forward, x0, u_seq)

    est_robot = DifferentialDriveModel(mass=10.0, tau_v=0.25)
    # Optimizer should automatically detect 5-state observable dimension
    optimizer = SystemIdentificationOptimizer(est_robot, dt=dt)
    res = optimizer.fit(
        x0=x0,
        u_seq=u_seq,
        y_target=y_target,
        adam_epochs=20,
        adam_lr=0.08,
        lbfgs_epochs=5,
        verbose=False,
    )

    assert res.final_rmse < res.initial_rmse
    assert res.status == "converged"


def test_optimizer_detached_inputs():
    """Verify fit() safely detaches inputs without throwing graph backward error."""
    torch.manual_seed(42)
    dt = 0.05
    sys = SimpleFirstOrderSystem(a=2.0, b=1.0)
    # Tensor created with computation history
    raw_x0 = torch.tensor([0.5], requires_grad=True)
    x0 = raw_x0 * 2.0
    u_seq = torch.ones(20, 1, dtype=torch.float64)
    y_target = torch.zeros(20, 1, dtype=torch.float64)

    opt = SystemIdentificationOptimizer(sys, dt=dt)
    # Should not raise RuntimeError about backwarding graph a second time
    res = opt.fit(x0, u_seq, y_target, adam_epochs=5, lbfgs_epochs=2, verbose=False)
    assert res.final_rmse > 0.0
