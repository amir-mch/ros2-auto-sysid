"""Safety envelope monitor and e-stop supervisor for robotic vehicle sysid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class GeofenceLimits:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    max_yaw_rate: float = 6.0        # [rad/s]
    max_lateral_accel: float = 12.0  # [m/s^2]
    max_speed: float = 6.0           # [m/s]


class SafetyMonitor:
    """Supervises real-time telemetry against boundary boxes and dynamic thresholds."""

    def __init__(self, limits: Optional[GeofenceLimits] = None) -> None:
        self.limits: GeofenceLimits = limits or GeofenceLimits()
        self.emergency_stop_triggered: bool = False
        self.trigger_reason: str = ""

    def check_state(
        self,
        px: float,
        py: float,
        vx: float,
        vy: float,
        omega_z: float,
    ) -> Tuple[bool, str]:
        """Verify whether the current vehicle state is within safe operating envelope.

        Returns:
            Tuple of (is_safe, violation_description).
        """
        if self.emergency_stop_triggered:
            return False, f"E-STOP Latched: {self.trigger_reason}"

        # Geofence boundary box check
        if not (self.limits.x_min <= px <= self.limits.x_max):
            return self._trigger(f"Geofence X violation: {px:.2f} m not in [{self.limits.x_min}, {self.limits.x_max}]")
        if not (self.limits.y_min <= py <= self.limits.y_max):
            return self._trigger(f"Geofence Y violation: {py:.2f} m not in [{self.limits.y_min}, {self.limits.y_max}]")

        # Dynamic stability checks
        if abs(omega_z) > self.limits.max_yaw_rate:
            return self._trigger(f"Yaw rate {omega_z:.2f} rad/s exceeds limit {self.limits.max_yaw_rate}")

        if abs(vx) > self.limits.max_speed:
            return self._trigger(f"Longitudinal speed {vx:.2f} m/s exceeds limit {self.limits.max_speed}")

        lat_accel = abs(vx * omega_z)
        if lat_accel > self.limits.max_lateral_accel:
            return self._trigger(f"Lateral acceleration {lat_accel:.2f} m/s^2 exceeds limit {self.limits.max_lateral_accel}")

        return True, "Nominal"

    def _trigger(self, reason: str) -> Tuple[bool, str]:
        self.emergency_stop_triggered = True
        self.trigger_reason = reason
        return False, reason

    def reset_estop(self) -> None:
        """Clear latched e-stop after vehicle has been secured."""
        self.emergency_stop_triggered = False
        self.trigger_reason = ""
