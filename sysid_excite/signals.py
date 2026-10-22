"""Excitation signal generators for dynamic system identification.

Generates persistently exciting control inputs (chirp sweeps, multi-steps,
slalom / figure-8 maneuvers) to excite vehicle lateral and longitudinal modes.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple
import numpy as np


def generate_chirp(
    time_array: np.ndarray,
    f0: float = 0.2,
    f1: float = 3.0,
    amplitude: float = 0.35,
    method: Literal["linear", "logarithmic"] = "linear",
) -> np.ndarray:
    """Generate frequency chirp sweep signal.

    Args:
        time_array: Monotonically increasing time coordinates [s].
        f0: Initial frequency at t=0 [Hz].
        f1: Final frequency at t=T [Hz].
        amplitude: Peak amplitude of the output wave.
        method: Sweep progression ('linear' or 'logarithmic').

    Returns:
        1D array of chirp signal values.
    """
    t_end = time_array[-1]
    if method == "linear":
        beta = (f1 - f0) / t_end
        phase = 2.0 * np.pi * (f0 * time_array + 0.5 * beta * time_array**2)
    elif method == "logarithmic":
        rate = np.log(f1 / f0) / t_end
        phase = 2.0 * np.pi * f0 * ((np.exp(rate * time_array) - 1.0) / rate)
    else:
        raise ValueError(f"Unknown sweep method: {method}")

    return amplitude * np.sin(phase)


def generate_step_sequence(
    time_array: np.ndarray,
    step_times: list[float],
    step_amplitudes: list[float],
) -> np.ndarray:
    """Generate multi-level step response command sequence."""
    signal = np.zeros_like(time_array)
    for t_step, amp in zip(step_times, step_amplitudes):
        signal[time_array >= t_step] = amp
    return signal


def generate_slalom_trajectory(
    duration: float = 10.0,
    dt: float = 0.02,
    base_velocity: float = 2.5,
    velocity_variation: float = 0.5,
    steering_max: float = 0.32,
    steering_freq: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate coordinated slalom / Figure-8 excitation sequence.

    Args:
        duration: Total excitation duration [s].
        dt: Sampling time interval [s].
        base_velocity: Nominal forward speed setpoint [m/s].
        velocity_variation: Amplitude of speed modulation [m/s].
        steering_max: Maximum front wheel angle [rad].
        steering_freq: Fundamental weaving frequency [Hz].

    Returns:
        Tuple of (time_array, control_sequence) with shape [T] and [T, 2].
    """
    n_samples = int(duration / dt)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # Longitudinal excitation: Base speed + multi-sine modulation
    v_cmd = (
        base_velocity
        + velocity_variation * np.sin(2.0 * np.pi * 0.25 * t)
        + 0.5 * velocity_variation * np.cos(2.0 * np.pi * 0.65 * t)
    )
    v_cmd = np.clip(v_cmd, 0.5, 5.0)

    # Steering excitation: Chirp sweep combined with harmonic slalom
    chirp_steer = generate_chirp(t, f0=0.15, f1=1.8, amplitude=0.6 * steering_max)
    harmonic_steer = 0.4 * steering_max * np.sin(2.0 * np.pi * steering_freq * t)
    delta_cmd = chirp_steer + harmonic_steer
    delta_cmd = np.clip(delta_cmd, -steering_max, steering_max)

    u_seq = np.column_stack([v_cmd, delta_cmd])
    return t, u_seq


def generate_figure8_trajectory(
    duration: float = 12.0,
    dt: float = 0.02,
    base_velocity: float = 2.0,
    velocity_variation: float = 0.4,
    steering_max: float = 0.32,
    loop_period: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate coordinated Figure-8 (Lemniscate) excitation maneuver.

    Executes alternating left and right high-dynamic turn cycles to persistently
    excite lateral dynamics symmetrically.

    Args:
        duration: Total maneuver duration [s].
        dt: Sampling interval [s].
        base_velocity: Nominal forward speed [m/s].
        velocity_variation: Speed modulation amplitude [m/s].
        steering_max: Maximum front wheel angle [rad].
        loop_period: Time duration to complete one full figure-8 loop [s].

    Returns:
        Tuple of (time_array, control_sequence) with shape [T] and [T, 2].
    """
    n_samples = int(duration / dt)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    omega = 2.0 * np.pi / loop_period
    base_steer = steering_max * np.sin(omega * t)
    chirp_super = 0.25 * steering_max * generate_chirp(t, f0=0.5, f1=2.5, amplitude=1.0)
    delta_cmd = np.clip(base_steer + chirp_super, -steering_max, steering_max)

    v_cmd = base_velocity - velocity_variation * np.abs(np.sin(omega * t))
    v_cmd = np.clip(v_cmd, 0.5, 5.0)

    u_seq = np.column_stack([v_cmd, delta_cmd])
    return t, u_seq
