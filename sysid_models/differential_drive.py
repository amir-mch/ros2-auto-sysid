"""Differential-drive mobile robot dynamical model.

Serves as an extension preset to demonstrate modular architecture flexibility.
"""

from __future__ import annotations

from typing import List, Tuple
import torch

from sysid_core.base_model import DifferentiableDynamicalSystem
from sysid_core.parameter import PhysicalParameter


class DifferentialDriveModel(DifferentiableDynamicalSystem):
    """Dynamic model of a differential-drive robot with motor lag and rolling resistance.

    State vector x (5-dim):
        x[0]: p_x     - World X position [m]
        x[1]: p_y     - World Y position [m]
        x[2]: psi     - Heading angle [rad]
        x[3]: v       - Forward linear velocity [m/s]
        x[4]: omega   - Angular velocity [rad/s]

    Control vector u (2-dim):
        u[0]: v_cmd     - Commanded linear velocity [m/s]
        u[1]: omega_cmd - Commanded angular velocity [rad/s]
    """

    STATE_NAMES = ["p_x", "p_y", "psi", "v", "omega"]
    CONTROL_NAMES = ["v_cmd", "omega_cmd"]

    def __init__(
        self,
        mass: float = 5.0,
        yaw_inertia: float = 0.15,
        track_width: float = 0.30,
        tau_v: float = 0.10,
        tau_omega: float = 0.08,
        drag_linear: float = 0.5,
        drag_angular: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(state_dim=5, control_dim=2)

        self.mass = self.register_physical_parameter(
            PhysicalParameter("mass", mass, (1.0, 30.0), unit="kg", dtype=dtype)
        )
        self.yaw_inertia = self.register_physical_parameter(
            PhysicalParameter("yaw_inertia", yaw_inertia, (0.01, 1.0), unit="kg*m^2", dtype=dtype)
        )
        self.track_width = self.register_physical_parameter(
            PhysicalParameter("track_width", track_width, (0.1, 1.0), unit="m", dtype=dtype)
        )
        self.tau_v = self.register_physical_parameter(
            PhysicalParameter("tau_v", tau_v, (0.01, 0.5), unit="s", dtype=dtype)
        )
        self.tau_omega = self.register_physical_parameter(
            PhysicalParameter("tau_omega", tau_omega, (0.01, 0.5), unit="s", dtype=dtype)
        )
        self.drag_linear = self.register_physical_parameter(
            PhysicalParameter("drag_linear", drag_linear, (0.01, 5.0), unit="N*s/m", dtype=dtype)
        )
        self.drag_angular = self.register_physical_parameter(
            PhysicalParameter("drag_angular", drag_angular, (0.001, 2.0), unit="N*m*s/rad", dtype=dtype)
        )

    @property
    def state_names(self) -> List[str]:
        return self.STATE_NAMES

    @property
    def control_names(self) -> List[str]:
        return self.CONTROL_NAMES

    def ode(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        psi = x[..., 2]
        v = x[..., 3]
        omega = x[..., 4]

        v_cmd = u[..., 0]
        omega_cmd = u[..., 1]

        # First-order motor lags with damping
        dv = (v_cmd - v) / self.tau_v() - (self.drag_linear() / self.mass()) * v
        domega = (omega_cmd - omega) / self.tau_omega() - (self.drag_angular() / self.yaw_inertia()) * omega

        # Global kinematics
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        dp_x = v * cos_psi
        dp_y = v * sin_psi
        dpsi = omega

        return torch.stack([dp_x, dp_y, dpsi, dv, domega], dim=-1)
