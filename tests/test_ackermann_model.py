"""Unit tests for Ackermann DynamicBicycleModel ODE dynamics."""

import torch
import pytest

from sysid_models.ackermann_dynamic import DynamicBicycleModel
from sysid_core.rk4_integrator import RK4Integrator


def test_ackermann_output_shape():
    """Verify ODE output dimension matches 7-state representation."""
    model = DynamicBicycleModel()
    t = torch.tensor(0.0, dtype=torch.float64)
    x = torch.zeros(7, dtype=torch.float64)
    u = torch.tensor([2.0, 0.1], dtype=torch.float64)

    dx = model.ode(t, x, u)
    assert dx.shape == torch.Size([7])
    assert not torch.isnan(dx).any()


def test_ackermann_standstill_numerical_stability():
    """Verify system does not encounter NaN or Inf at standstill (v_x = 0)."""
    model = DynamicBicycleModel()
    t = torch.tensor(0.0, dtype=torch.float64)
    # Complete standstill with zero velocities
    x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    u = torch.tensor([0.0, 0.0], dtype=torch.float64)

    dx = model.ode(t, x, u)
    assert not torch.isnan(dx).any()
    assert not torch.isinf(dx).any()
    # At zero command and zero state, all derivatives should be zero
    assert torch.allclose(dx, torch.zeros_like(dx), atol=1e-5)


def test_ackermann_straight_acceleration():
    """Verify straight line command increases longitudinal velocity without lateral drift."""
    model = DynamicBicycleModel()
    integrator = RK4Integrator(dt=0.02, hold_mode="linear", n_substeps=2)

    x0 = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=torch.float64)
    # Command constant 3.0 m/s with zero steering
    u_seq = torch.tensor([[3.0, 0.0]] * 50, dtype=torch.float64)

    traj = integrator.rollout(model.forward, x0, u_seq)
    final_state = traj[-1]

    # Forward velocity must have increased
    assert final_state[3] > 0.5
    # Lateral velocity and yaw rate must remain negligible on straight path
    assert abs(final_state[4].item()) < 1e-4
    assert abs(final_state[5].item()) < 1e-4
    # Lateral position y must remain zero
    assert abs(final_state[1].item()) < 1e-4


def test_ackermann_cornering_response():
    """Verify positive steering angle induces positive yaw rate and curved trajectory."""
    model = DynamicBicycleModel()
    integrator = RK4Integrator(dt=0.02, hold_mode="linear", n_substeps=2)

    x0 = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    # Command constant forward velocity and positive steering (left turn)
    u_seq = torch.tensor([[2.0, 0.25]] * 60, dtype=torch.float64)

    traj = integrator.rollout(model.forward, x0, u_seq)
    final_state = traj[-1]

    # Positive steering induces positive yaw rate and heading
    assert final_state[5] > 0.0   # omega_z > 0
    assert final_state[2] > 0.0   # psi > 0
    assert final_state[1] > 0.0   # Y position curved upward > 0


def test_ackermann_observation_projection():
    """Verify observe() projects 7D internal state to 6D telemetry vector."""
    model = DynamicBicycleModel()
    x = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.1, 0.2, 0.15], dtype=torch.float64)
    y = model.observe(x)

    assert y.shape == torch.Size([6])
    assert torch.allclose(y, x[:6])


def test_ackermann_reverse_driving_stability():
    """Verify reverse driving remains numerically stable without wild yaw divergence."""
    model = DynamicBicycleModel()
    integrator = RK4Integrator(dt=0.02, hold_mode="linear", n_substeps=2)

    x0 = torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    # Reverse command at -1.0 m/s with small steering angle
    u_seq = torch.tensor([[-1.0, 0.05]] * 100, dtype=torch.float64)

    traj = integrator.rollout(model.forward, x0, u_seq)
    final_state = traj[-1]

    # Vehicle moved backwards along X
    assert final_state[0] < -1.5
    # Longitudinal speed remains negative and near -1.0 m/s
    assert final_state[3] < -0.8
    # Yaw rate remains finite and bounded (not wildly diverging to >30 rad/s)
    assert abs(final_state[5].item()) < 2.0
    # Heading remains bounded and reasonable
    assert abs(final_state[2].item()) < 1.5


def test_ackermann_saturating_tire_model():
    """Verify smooth saturating tire model bounds lateral forces and propagates gradients."""
    model = DynamicBicycleModel(tire_model_type="saturating", friction_coeff=0.85)
    t = torch.tensor(0.0, dtype=torch.float64)
    # Extreme slip condition (large lateral velocity vy=4.0 at vx=2.0)
    x = torch.tensor([0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0], dtype=torch.float64)
    u = torch.tensor([2.0, 0.0], dtype=torch.float64)

    dx = model.ode(t, x, u)
    assert not torch.isnan(dx).any()
    assert not torch.isinf(dx).any()

    # Verify autograd gradient flows into friction_coeff
    loss = dx.sum()
    loss.backward()
    assert model.friction_coeff.raw_param.grad is not None
    assert model.friction_coeff.raw_param.grad.abs().item() > 0.0


def test_ackermann_learn_geometry():
    """Verify l_f and l_r axle distances can be calibrated when learn_geometry=True."""
    model = DynamicBicycleModel(learn_geometry=True, l_f=0.15, l_r=0.18)
    t = torch.tensor(0.0, dtype=torch.float64)
    x = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.2, 0.5, 0.1], dtype=torch.float64)
    u = torch.tensor([2.0, 0.1], dtype=torch.float64)

    dx = model.ode(t, x, u)
    loss = dx[5].pow(2)  # domega_z
    loss.backward()

    assert model.l_f.raw_param.grad is not None
    assert model.l_r.raw_param.grad is not None
