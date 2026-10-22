"""Base interface for differentiable continuous-time dynamical systems.

Enforces ODE structure and physical parameter tracking for grey-box sysid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from sysid_core.parameter import PhysicalParameter


class DifferentiableDynamicalSystem(nn.Module, ABC):
    """Abstract base class for grey-box continuous-time dynamical models.

    Subclasses implement the continuous-time ordinary differential equation:
        dx/dt = f(t, x, u; theta)
    where theta is a collection of learnable PhysicalParameter instances.
    """

    def __init__(self, state_dim: int, control_dim: int) -> None:
        super().__init__()
        self.state_dim: int = state_dim
        self.control_dim: int = control_dim
        self._param_registry: Dict[str, PhysicalParameter] = {}

    def register_physical_parameter(self, param: PhysicalParameter) -> PhysicalParameter:
        """Register a bounded parameter as a torch sub-module and index it."""
        self.add_module(param.name, param)
        self._param_registry[param.name] = param
        return param

    @property
    @abstractmethod
    def state_names(self) -> List[str]:
        """Names of state variables."""
        pass

    @property
    @abstractmethod
    def control_names(self) -> List[str]:
        """Names of control input variables."""
        pass

    @abstractmethod
    def ode(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute state derivative dx/dt.

        Args:
            t: Scalar or batch of time coordinates [s].
            x: State tensor with shape (..., state_dim).
            u: Control tensor with shape (..., control_dim).

        Returns:
            State derivative dx/dt with shape (..., state_dim).
        """
        pass

    def forward(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Evaluate continuous-time ODE derivative."""
        return self.ode(t, x, u)

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """Map internal state to observable state vector.

        Default returns x directly. Subclasses override if internal state has
        unobservable actuator states.
        """
        return x

    @property
    def observable_dim(self) -> int:
        """Dimension of the observed state vector."""
        return self.state_dim

    @property
    def observable_names(self) -> List[str]:
        """Names of observed states."""
        return self.state_names

    def get_physical_parameters(self) -> Dict[str, float]:
        """Extract dictionary of current physical values."""
        return {name: p.value for name, p in self._param_registry.items()}

    def get_physical_parameter_tensors(self) -> Dict[str, torch.Tensor]:
        """Extract dictionary of differentiable parameter tensors."""
        return {name: p() for name, p in self._param_registry.items()}

    def prior_regularization_loss(self) -> torch.Tensor:
        """Total MAP regularization penalty across all registered parameters."""
        losses = [p.prior_loss() for p in self._param_registry.values()]
        if not losses:
            return torch.tensor(0.0, dtype=torch.float64)
        return torch.stack(losses).sum()
