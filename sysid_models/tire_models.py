"""Tire lateral force models for ground vehicle dynamics.

Provides linear and smooth saturation models suitable for gradient-based sysid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearTireModel(nn.Module):
    """Linear cornering stiffness tire model.

    Computes lateral tire force:
        F_y = C_alpha * alpha
    where C_alpha is cornering stiffness [N/rad] and alpha is slip angle [rad].
    """

    def forward(self, cornering_stiffness: torch.Tensor, slip_angle: torch.Tensor) -> torch.Tensor:
        return cornering_stiffness * slip_angle


class SmoothSaturatingTireModel(nn.Module):
    """Smooth saturating tire model based on hyperbolic tangent.

    Computes lateral tire force:
        F_y = F_max * tanh((C_alpha * alpha) / F_max)
    Yields linear behavior at small slip angles (dF_y/dalpha = C_alpha at alpha=0)
    and asymptotically saturates to F_max = mu * F_z.
    """

    def forward(
        self,
        cornering_stiffness: torch.Tensor,
        slip_angle: torch.Tensor,
        normal_force: torch.Tensor,
        friction_coeff: torch.Tensor,
    ) -> torch.Tensor:
        f_max = torch.clamp(friction_coeff * normal_force, min=1e-3)
        linear_force = cornering_stiffness * slip_angle
        return f_max * torch.tanh(linear_force / f_max)
