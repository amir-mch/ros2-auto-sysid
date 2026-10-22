"""Physical parameter abstractions with differentiable domain constraints.

Reparameterizes constrained physical variables into unconstrained latent tensors
so gradient descent cannot produce non-physical values (e.g. negative mass).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PhysicalParameter(nn.Module):
    """Learnable physical parameter mapped through smooth bounded transformations.

    Attributes:
        name: Human-readable parameter identifier.
        unit: Physical unit string (e.g. 'kg', 'N/rad').
        lower_bound: Minimum physically allowable value.
        upper_bound: Maximum physically allowable value.
        raw_param: Unconstrained latent tensor optimized by gradient descent.
    """

    def __init__(
        self,
        name: str,
        initial_value: float,
        bounds: Tuple[float, float],
        unit: str = "",
        prior_std: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if bounds[0] >= bounds[1]:
            raise ValueError(f"Lower bound {bounds[0]} must be strictly less than upper bound {bounds[1]}")
        if not (bounds[0] <= initial_value <= bounds[1]):
            raise ValueError(
                f"Initial value {initial_value} for '{name}' is outside bounds [{bounds[0]}, {bounds[1]}]"
            )

        self.name: str = name
        self.unit: str = unit
        self.lower_bound: float = float(bounds[0])
        self.upper_bound: float = float(bounds[1])
        self.prior_mean: float = float(initial_value)
        self.prior_std: Optional[float] = float(prior_std) if prior_std is not None else None

        # Solve for raw latent value such that sigmoid(raw) produces initial_value
        span = self.upper_bound - self.lower_bound
        normalized = (initial_value - self.lower_bound) / span
        clamped = max(1e-6, min(1.0 - 1e-6, normalized))
        raw_init = math.log(clamped / (1.0 - clamped))

        self.raw_param = nn.Parameter(
            torch.tensor(raw_init, dtype=dtype, device=device, requires_grad=True)
        )

    def forward(self) -> torch.Tensor:
        """Evaluate the physical parameter within [lower_bound, upper_bound]."""
        span = self.upper_bound - self.lower_bound
        return self.lower_bound + span * torch.sigmoid(self.raw_param)

    @property
    def value(self) -> float:
        """Current scalar value as Python float."""
        return float(self.forward().detach().cpu().item())

    def set_value(self, target_value: float) -> None:
        """Update latent parameter to match a specific physical target."""
        if not (self.lower_bound <= target_value <= self.upper_bound):
            raise ValueError(
                f"Target value {target_value} out of bounds [{self.lower_bound}, {self.upper_bound}]"
            )
        span = self.upper_bound - self.lower_bound
        normalized = (target_value - self.lower_bound) / span
        clamped = max(1e-6, min(1.0 - 1e-6, normalized))
        raw_val = math.log(clamped / (1.0 - clamped))
        with torch.no_grad():
            self.raw_param.copy_(torch.tensor(raw_val, dtype=self.raw_param.dtype, device=self.raw_param.device))

    def prior_loss(self) -> torch.Tensor:
        """Quadratic regularization penalty against prior belief."""
        if self.prior_std is None or self.prior_std <= 0.0:
            return torch.tensor(0.0, dtype=self.raw_param.dtype, device=self.raw_param.device)
        curr = self.forward()
        diff = (curr - self.prior_mean) / self.prior_std
        return 0.5 * diff.pow(2)

    def __repr__(self) -> str:
        return (
            f"PhysicalParameter(name='{self.name}', value={self.value:.5g} {self.unit}, "
            f"bounds=[{self.lower_bound:.5g}, {self.upper_bound:.5g}])"
        )
