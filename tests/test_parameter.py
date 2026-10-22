"""Unit tests for PhysicalParameter domain bounds and reparameterization."""

import pytest
import torch

from sysid_core.parameter import PhysicalParameter


def test_parameter_bounds_enforcement():
    """Verify parameter cannot exceed prescribed bounds even with extreme raw logits."""
    lower = 2.0
    upper = 6.0
    param = PhysicalParameter("test_mass", initial_value=3.0, bounds=(lower, upper))

    # Test extreme positive latent
    param.raw_param.data.fill_(1000.0)
    val_high = param().item()
    assert val_high <= upper
    assert abs(val_high - upper) < 1e-4

    # Test extreme negative latent
    param.raw_param.data.fill_(-1000.0)
    val_low = param().item()
    assert val_low >= lower
    assert abs(val_low - lower) < 1e-4


def test_parameter_initialization_accuracy():
    """Verify initial physical value matches the configured initial value."""
    initial_val = 75.0
    param = PhysicalParameter("c_f", initial_value=initial_val, bounds=(10.0, 200.0))
    assert abs(param.value - initial_val) < 1e-5


def test_parameter_invalid_bounds():
    """Verify ValueError is raised on invalid bound definitions."""
    with pytest.raises(ValueError):
        PhysicalParameter("bad_bounds", initial_value=5.0, bounds=(10.0, 2.0))

    with pytest.raises(ValueError):
        PhysicalParameter("out_of_bounds", initial_value=1.0, bounds=(2.0, 10.0))


def test_parameter_set_value():
    """Verify set_value correctly re-computes raw latent."""
    param = PhysicalParameter("steering_tau", initial_value=0.08, bounds=(0.01, 0.40))
    target = 0.20
    param.set_value(target)
    assert abs(param.value - target) < 1e-5


def test_parameter_gradient_flow():
    """Verify autograd computes gradients through sigmoid reparameterization."""
    param = PhysicalParameter("motor_gain", initial_value=20.0, bounds=(5.0, 50.0))
    val = param()
    loss = (val - 30.0).pow(2)
    loss.backward()

    assert param.raw_param.grad is not None
    assert param.raw_param.grad.abs().item() > 0.0


def test_parameter_prior_loss():
    """Verify quadratic prior regularization loss behavior."""
    mean = 10.0
    std = 2.0
    param = PhysicalParameter("p", initial_value=mean, bounds=(1.0, 20.0), prior_std=std)

    # At prior mean, penalty is zero
    assert abs(param.prior_loss().item()) < 1e-6

    # Offset by 1 std deviation: 0.5 * ((mean + std - mean) / std)^2 = 0.5
    param.set_value(mean + std)
    assert abs(param.prior_loss().item() - 0.5) < 1e-4
