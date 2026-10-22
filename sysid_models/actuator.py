"""Actuator response dynamics models.

Models the first-order lag, bandwidth limitations, and rate constraints of
low-level servos (steering) and electronic speed controllers (ESC).
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn

from sysid_core.parameter import PhysicalParameter


class FirstOrderActuator(nn.Module):
    """First-order low-pass filter representing actuator transient response.

    Continuous-time state equation:
        da/dt = (u_cmd - a) / tau

    where:
        a: Actual actuator state (e.g. steering angle, drivetrain velocity).
        u_cmd: Commanded input signal.
        tau: Effective time constant [s].
    """

    def __init__(
        self,
        name: str,
        initial_tau: float = 0.08,
        bounds_tau: Tuple[float, float] = (0.005, 0.5),
        rate_limit: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.name: str = name
        self.rate_limit: Optional[float] = float(rate_limit) if rate_limit is not None else None
        self.tau = PhysicalParameter(
            name=f"{name}_tau",
            initial_value=initial_tau,
            bounds=bounds_tau,
            unit="s",
            prior_std=0.05,
            dtype=dtype,
        )

    def derivative(self, actual: torch.Tensor, commanded: torch.Tensor) -> torch.Tensor:
        """Compute time derivative da/dt.

        Args:
            actual: Current internal actuator state.
            commanded: Commanded reference setpoint.

        Returns:
            Rate of change da/dt.
        """
        tau_val = self.tau()
        # Ensure numerical safety with lower epsilon on tau
        safe_tau = torch.clamp(tau_val, min=1e-4)
        rate = (commanded - actual) / safe_tau

        if self.rate_limit is not None and self.rate_limit > 0.0:
            # Smooth rate saturation using tanh to maintain gradient continuity
            rl = self.rate_limit
            rate = rl * torch.tanh(rate / rl)

        return rate

    @property
    def time_constant(self) -> float:
        """Current time constant in seconds."""
        return self.tau.value
