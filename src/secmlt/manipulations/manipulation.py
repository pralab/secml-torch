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
        self._domain_constraints = domain_constraints
        self._perturbation_constraints = perturbation_constraints

    @property
    def domain_constraints(self) -> list[Constraint]:
        """
        Get the domain constraints for the manipulation.

        Returns
        -------
        list[Constraint]
            List of domain constraints for the manipulation.
        """
        return self._domain_constraints

    @domain_constraints.setter
    def domain_constraints(self, domain_constraints: list[Constraint]) -> None:
        self._domain_constraints = domain_constraints

    @property
    def perturbation_constraints(self) -> list[Constraint]:
        """
        Get the perturbation constraints for the manipulation.

        Returns
        -------
        list[Constraint]
            List of perturbation constraints for the manipulation.
        """
        return self._perturbation_constraints

    @perturbation_constraints.setter
    def perturbation_constraints(
        self, perturbation_constraints: list[Constraint]
    ) -> None:
        self._perturbation_constraints = perturbation_constraints

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

    @abstractmethod
    def _invert_manipulation(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Invert the manipulation to re-obtain the perturbation.

        Does not need to be implemented if the manipulation does
        not allow inversion (defaults to None).

        Parameters
        ----------
        x : torch.Tensor
            Original input samples.
        x_adv : torch.Tensor
            Perturbed samples.

        Returns
        -------
        torch.Tensor or None
            Perturbation obtained by inverting the manipulation.
            If the manipulation does not allow inversion, returns None.
        """
        return None

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
        # apply the domain constraints to the perturbed sample
        x_adv.data = self._apply_domain_constraints(x_adv.data)
        # obtain delta after applying the domain constraint
        in_domain_delta = self._invert_manipulation(x, x_adv)
        if in_domain_delta is not None:
            delta.data = in_domain_delta.data
        return x_adv, delta


class AdditiveManipulation(Manipulation):
    """Additive manipulation for input data."""

    def _apply_manipulation(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + delta, delta

    def _invert_manipulation(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor
        ) -> torch.Tensor:
        return x_adv - x

