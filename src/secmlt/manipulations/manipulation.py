"""Manipulations for perturbing input samples."""

from abc import ABC, abstractmethod

import torch
from secmlt.optimization.constraints import Constraint


class Manipulation(ABC):
    """Abstract class for manipulations."""

    def __init__(
        self,
        domain_constraints: list[Constraint],
        perturbation_constraints: list[Constraint],
    ) -> None:
        """
        Create manipulation object.

        Parameters
        ----------
        domain_constraints : list[Constraint]
            Constraints for the domain bounds (x_adv).
        perturbation_constraints : list[Constraint]
            Constraints for the perturbation (delta).
        """
        self.domain_constraints = domain_constraints
        self.perturbation_constraints = perturbation_constraints

    def _apply_domain_constraints(self, x: torch.Tensor) -> torch.Tensor:
        for constraint in self.domain_constraints:
            x = constraint(x)
        return x

    def _apply_perturbation_constraints(self, delta: torch.Tensor) -> torch.Tensor:
        for constraint in self.perturbation_constraints:
            delta = constraint(delta)
        return delta

    @abstractmethod
    def _apply_manipulation(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the manipulation.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.
        delta : torch.Tensor
            Manipulation to apply.

        Returns
        -------
        torch.Tensor
            Perturbed samples.
        """
        ...

    def __call__(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the manipulation to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        delta : torch.Tensor
            Perturbation to apply.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Perturbed data and perturbation after the
            application of constraints.
        """
        delta.data = self._apply_perturbation_constraints(delta.data)
        x_adv, delta = self._apply_manipulation(x, delta)
        x_adv.data = self._apply_domain_constraints(x_adv.data)
        return x_adv, delta


class AdditiveManipulation(Manipulation):
    """Additive manipulation for input data."""

    def _apply_manipulation(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + delta, delta
