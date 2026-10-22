"""Dynamic Bicycle Model with actuator dynamics for Ackermann-steered vehicles.

Implements differentiable continuous-time equations of motion for autonomous
vehicles (e.g. 1:10 scale F1TENTH / RoboRacer) incorporating lateral tire slip,
yaw moment of inertia, drivetrain dynamics, and steering servo lag.
Supports both linear and saturating tire models, with numerical stability across
low-speed standstill and reverse driving.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import torch

from sysid_core.base_model import DifferentiableDynamicalSystem
from sysid_core.parameter import PhysicalParameter
from sysid_models.actuator import FirstOrderActuator
from sysid_models.tire_models import LinearTireModel, SmoothSaturatingTireModel


class DynamicBicycleModel(DifferentiableDynamicalSystem):
    """Grey-box dynamic bicycle model with 1st-order steering lag.

    State vector x (7-dim):
        x[0]: p_x    - Inertial X position in world frame [m]
        x[1]: p_y    - Inertial Y position in world frame [m]
        x[2]: psi    - Yaw heading angle [rad]
        x[3]: v_x    - Longitudinal body velocity [m/s]
        x[4]: v_y    - Lateral body velocity [m/s]
        x[5]: omega_z- Yaw angular velocity / yaw rate [rad/s]
        x[6]: delta  - Actual steering angle at front wheels [rad]

    Control vector u (2-dim):
        u[0]: v_cmd     - Commanded longitudinal velocity setpoint [m/s]
        u[1]: delta_cmd - Commanded front steering angle [rad]

    Observable vector y (6-dim by default, from Odometry + IMU):
        [p_x, p_y, psi, v_x, v_y, omega_z]
    """

    STATE_NAMES = ["p_x", "p_y", "psi", "v_x", "v_y", "omega_z", "delta"]
    OBSERVABLE_NAMES = ["p_x", "p_y", "psi", "v_x", "v_y", "omega_z"]
    CONTROL_NAMES = ["v_cmd", "delta_cmd"]

    def __init__(
        self,
        # Nominal / Prior values for 1:10 scale F1TENTH platform
        mass: float = 3.4,
        yaw_inertia: float = 0.045,
        l_f: float = 0.16,
        l_r: float = 0.17,
        c_f: float = 75.0,
        c_r: float = 85.0,
        motor_gain: float = 20.0,
        drag_coeff: float = 0.25,
        steering_tau: float = 0.07,
        friction_coeff: float = 0.85,
        # Bounds for physically safe reparameterization
        bounds_mass: Tuple[float, float] = (1.5, 7.0),
        bounds_yaw_inertia: Tuple[float, float] = (0.01, 0.20),
        bounds_c_f: Tuple[float, float] = (15.0, 250.0),
        bounds_c_r: Tuple[float, float] = (15.0, 250.0),
        bounds_motor_gain: Tuple[float, float] = (2.0, 80.0),
        bounds_drag_coeff: Tuple[float, float] = (0.01, 2.0),
        bounds_steering_tau: Tuple[float, float] = (0.01, 0.40),
        bounds_friction_coeff: Tuple[float, float] = (0.20, 1.80),
        learn_geometry: bool = False,
        tire_model_type: Literal["linear", "saturating"] = "linear",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(state_dim=7, control_dim=2)
        self.tire_model_type: Literal["linear", "saturating"] = tire_model_type

        # Core vehicle inertial properties
        self.mass = self.register_physical_parameter(
            PhysicalParameter("mass", mass, bounds_mass, unit="kg", prior_std=0.5, dtype=dtype)
        )
        self.yaw_inertia = self.register_physical_parameter(
            PhysicalParameter("yaw_inertia", yaw_inertia, bounds_yaw_inertia, unit="kg*m^2", prior_std=0.01, dtype=dtype)
        )

        # Tire cornering stiffness
        self.c_f = self.register_physical_parameter(
            PhysicalParameter("c_f", c_f, bounds_c_f, unit="N/rad", prior_std=15.0, dtype=dtype)
        )
        self.c_r = self.register_physical_parameter(
            PhysicalParameter("c_r", c_r, bounds_c_r, unit="N/rad", prior_std=15.0, dtype=dtype)
        )

        # Drivetrain and aerodynamic drag
        self.motor_gain = self.register_physical_parameter(
            PhysicalParameter("motor_gain", motor_gain, bounds_motor_gain, unit="N*s/m", prior_std=5.0, dtype=dtype)
        )
        self.drag_coeff = self.register_physical_parameter(
            PhysicalParameter("drag_coeff", drag_coeff, bounds_drag_coeff, unit="N*s^2/m^2", prior_std=0.1, dtype=dtype)
        )

        # Steering actuator dynamics
        self.steering_actuator = FirstOrderActuator(
            name="steering",
            initial_tau=steering_tau,
            bounds_tau=bounds_steering_tau,
            rate_limit=8.0,  # Max ~8 rad/s servo sweep rate
            dtype=dtype,
        )
        self.register_physical_parameter(self.steering_actuator.tau)

        # Tire friction coefficient for saturating tire model
        if self.tire_model_type == "saturating":
            self.friction_coeff = self.register_physical_parameter(
                PhysicalParameter("friction_coeff", friction_coeff, bounds_friction_coeff, unit="-", prior_std=0.1, dtype=dtype)
            )
            self.tire_model = SmoothSaturatingTireModel()
        else:
            self.tire_model = LinearTireModel()

        # Axle distances (fixed by default from CAD measurements)
        self.learn_geometry = learn_geometry
        if learn_geometry:
            self.l_f = self.register_physical_parameter(
                PhysicalParameter("l_f", l_f, (0.08, 0.30), unit="m", prior_std=0.02, dtype=dtype)
            )
            self.l_r = self.register_physical_parameter(
                PhysicalParameter("l_r", l_r, (0.08, 0.30), unit="m", prior_std=0.02, dtype=dtype)
            )
        else:
            self.register_buffer("l_f_const", torch.tensor(l_f, dtype=dtype))
            self.register_buffer("l_r_const", torch.tensor(l_r, dtype=dtype))

        self.gravity = 9.81  # [m/s^2]

        # Regularization constants for numerical stability at near-zero velocity
        self.v_regularization = 0.15  # [m/s] Prevents division by zero in slip angle calculation
        self.v_blend = 0.20           # [m/s] Smoothly scales lateral tire forces at low speed

    @property
    def lf_val(self) -> torch.Tensor:
        return self.l_f() if self.learn_geometry else self.l_f_const

    @property
    def lr_val(self) -> torch.Tensor:
        return self.l_r() if self.learn_geometry else self.l_r_const

    @property
    def state_names(self) -> List[str]:
        return self.STATE_NAMES

    @property
    def control_names(self) -> List[str]:
        return self.CONTROL_NAMES

    @property
    def observable_names(self) -> List[str]:
        return self.OBSERVABLE_NAMES

    @property
    def observable_dim(self) -> int:
        return 6

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """Extract observable states [p_x, p_y, psi, v_x, v_y, omega_z]."""
        return x[..., :6]

    def ode(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute continuous-time state derivatives dx/dt.

        Args:
            t: Current time [s].
            x: State tensor [..., 7] -> (p_x, p_y, psi, v_x, v_y, omega_z, delta).
            u: Control tensor [..., 2] -> (v_cmd, delta_cmd).

        Returns:
            dx/dt tensor [..., 7].
        """
        # Unpack states
        psi = x[..., 2]
        v_x = x[..., 3]
        v_y = x[..., 4]
        omega_z = x[..., 5]
        delta = x[..., 6]

        # Unpack control commands
        v_cmd = u[..., 0]
        delta_cmd = u[..., 1]

        # Differentiable parameters
        m = self.mass()
        i_z = self.yaw_inertia()
        c_f = self.c_f()
        c_r = self.c_r()
        k_m = self.motor_gain()
        c_d = self.drag_coeff()
        lf = self.lf_val
        lr = self.lr_val

        # Numerically stable forward velocity for slip angle calculation
        v_x_safe = torch.sqrt(v_x * v_x + self.v_regularization * self.v_regularization)

        # Smooth, continuous direction factor (tanh(v_x / 0.05) preserves C^\infty differentiability)
        smooth_sign = torch.tanh(v_x / 0.05)

        # Tire slip angles
        alpha_f = smooth_sign * delta - torch.atan2(v_y + lf * omega_z, v_x_safe)
        alpha_r = - torch.atan2(v_y - lr * omega_z, v_x_safe)

        # Smooth low-speed blending to prevent oscillatory forces when stationary
        # Uses abs(v_x) so scaling factor is strictly non-negative in [0, 1]
        blend_factor = torch.tanh(torch.abs(v_x) / self.v_blend)

        # Lateral tire forces (linear or saturating)
        if self.tire_model_type == "saturating":
            mu = self.friction_coeff()
            total_wheelbase = lf + lr
            f_z_f = m * self.gravity * (lr / total_wheelbase)
            f_z_r = m * self.gravity * (lf / total_wheelbase)
            f_y_f = blend_factor * self.tire_model(c_f, alpha_f, f_z_f, mu)
            f_y_r = blend_factor * self.tire_model(c_r, alpha_r, f_z_r, mu)
        else:
            f_y_f = blend_factor * self.tire_model(c_f, alpha_f)
            f_y_r = blend_factor * self.tire_model(c_r, alpha_r)

        # Longitudinal drivetrain drive and aerodynamic resistance
        f_drive = k_m * (v_cmd - v_x)
        f_drag = c_d * v_x * torch.abs(v_x)
        f_x = f_drive - f_drag

        # Dynamic equations of motion in body frame
        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)

        # d(v_x)/dt = 1/m * (F_x - F_yf * sin(delta)) + v_y * omega_z
        dv_x = (1.0 / m) * (f_x - f_y_f * sin_delta) + v_y * omega_z

        # d(v_y)/dt = 1/m * (F_yr + F_yf * cos(delta)) - v_x * omega_z
        dv_y = (1.0 / m) * (f_y_r + f_y_f * cos_delta) - v_x * omega_z

        # d(omega_z)/dt = 1/I_z * (lf * F_yf * cos(delta) - lr * F_yr)
        domega_z = (1.0 / i_z) * (lf * f_y_f * cos_delta - lr * f_y_r)

        # Steering actuator rate
        ddelta = self.steering_actuator.derivative(delta, delta_cmd)

        # Kinematic coordinate transformation to world frame
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        dp_x = v_x * cos_psi - v_y * sin_psi
        dp_y = v_x * sin_psi + v_y * cos_psi
        dpsi = omega_z

        return torch.stack([dp_x, dp_y, dpsi, dv_x, dv_y, domega_z, ddelta], dim=-1)
