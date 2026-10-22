"""Unit tests for excitation signal generators and safety monitor."""

import numpy as np

from sysid_excite.signals import (
    generate_chirp,
    generate_step_sequence,
    generate_slalom_trajectory,
    generate_figure8_trajectory,
)
from sysid_excite.safety_monitor import SafetyMonitor, GeofenceLimits


def test_chirp_bounds():
    """Verify chirp signals do not exceed peak amplitude."""
    t = np.linspace(0, 5, 500)
    amplitude = 0.35
    sig = generate_chirp(t, f0=0.1, f1=2.0, amplitude=amplitude)

    assert sig.shape == (500,)
    assert np.max(np.abs(sig)) <= amplitude + 1e-6


def test_step_sequence():
    """Verify step sequence matches prescribed step timing and amplitudes."""
    t = np.linspace(0, 4, 401)
    steps = [1.0, 2.5]
    amps = [0.2, -0.2]
    sig = generate_step_sequence(t, steps, amps)

    assert sig[t < 1.0].tolist() == [0.0] * np.sum(t < 1.0)
    assert np.allclose(sig[(t >= 1.0) & (t < 2.5)], 0.2)
    assert np.allclose(sig[t >= 2.5], -0.2)


def test_slalom_trajectory():
    """Verify slalom trajectory outputs proper shapes and velocity/steering limits."""
    t, u = generate_slalom_trajectory(duration=6.0, dt=0.02, steering_max=0.30)

    assert len(t) == len(u)
    assert u.shape[1] == 2
    # Commanded forward velocity must be positive
    assert np.all(u[:, 0] > 0.0)
    # Steering command must stay within max steering limit
    assert np.all(np.abs(u[:, 1]) <= 0.30 + 1e-5)


def test_safety_monitor_nominal_and_violations():
    """Verify safety monitor approves nominal states and catches boundary / stability violations."""
    limits = GeofenceLimits(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, max_yaw_rate=4.0)
    monitor = SafetyMonitor(limits)

    # Nominal state
    is_safe, msg = monitor.check_state(px=1.0, py=2.0, vx=2.0, vy=0.1, omega_z=1.5)
    assert is_safe
    assert msg == "Nominal"

    # Geofence violation
    is_safe, msg = monitor.check_state(px=6.5, py=2.0, vx=2.0, vy=0.1, omega_z=1.5)
    assert not is_safe
    assert "Geofence X violation" in msg

    # Once triggered, monitor latches e-stop
    is_safe_latched, _ = monitor.check_state(px=0.0, py=0.0, vx=0.0, vy=0.0, omega_z=0.0)
    assert not is_safe_latched

    # Reset e-stop
    monitor.reset_estop()

    # Dynamic yaw rate violation
    is_safe_yaw, msg_yaw = monitor.check_state(px=0.0, py=0.0, vx=1.0, vy=0.0, omega_z=5.5)
    assert not is_safe_yaw
    assert "Yaw rate" in msg_yaw


def test_figure8_trajectory():
    """Verify Figure-8 maneuver satisfies duration, positive speed, and steering bounds."""
    duration = 10.0
    dt = 0.02
    t, u = generate_figure8_trajectory(duration=duration, dt=dt, steering_max=0.30)

    assert len(t) == int(duration / dt)
    assert u.shape == (len(t), 2)
    # Forward velocity must remain positive throughout maneuver
    assert np.all(u[:, 0] > 0.0)
    # Front wheel steering must respect limits
    assert np.all(np.abs(u[:, 1]) <= 0.30 + 1e-5)
    # Steering command must have both positive and negative extremes (alternating loops)
    assert np.min(u[:, 1]) < -0.15
    assert np.max(u[:, 1]) > 0.15
