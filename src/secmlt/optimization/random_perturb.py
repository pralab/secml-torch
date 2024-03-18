"""Random pertubations in Lp balls."""

from abc import ABC, abstractmethod

import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.optimization.constraints import (
    L0Constraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
    LpConstraint,
)
from torch.distributions.laplace import Laplace


class RandomPerturbBase(ABC):
    """Class implementing the random perturbations in Lp balls."""

    def __init__(self, epsilon: float) -> None:
        """
        Create random perturbation object.

        Parameters
        ----------
        epsilon : float
            Constraint radius.
        """
        self.epsilon = epsilon

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the perturbations for the given samples.

        Parameters
        ----------
        x : torch.Tensor
            Input samples to perturb.

        Returns
        -------
        torch.Tensor
            Perturbations (to apply) to the given samples.
        """
        perturbations = self.get_perturb(x)
        return self._constraint(
            radius=self.epsilon,
            center=torch.zeros_like(perturbations),
        ).project(perturbations)

    @abstractmethod
    def get_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate random perturbation for the Lp norm.

        Parameters
        ----------
        x : torch.Tensor
            Input samples to perturb.
        """
        ...

    @abstractmethod
    def _constraint(self) -> LpConstraint:
        ...


class RandomPerturbLinf(RandomPerturbBase):
    """Random Perturbations for Linf norm."""

    def get_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate random perturbation for the Linf norm.

        Parameters
        ----------
        x : torch.Tensor
            Input samples to perturb.

        Returns
        -------
        torch.Tensor
            Perturbed samples.
        """
        return torch.randn_like(x)

    @property
    def _constraint(self) -> type[LInfConstraint]:
        return LInfConstraint


class RandomPerturbL1(RandomPerturbBase):
    """Random Perturbations for L1 norm."""

    def get_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate random perturbation for the L1 norm.

        Parameters
        ----------
        x : torch.Tensor
            Input samples to perturb.

        Returns
        -------
        torch.Tensor
            Perturbed samples.
        """
        s = Laplace(loc=0, scale=1)
        return s.sample(x.shape)

    @property
    def _constraint(self) -> type[L1Constraint]:
        return L1Constraint


class RandomPerturbL2(RandomPerturbBase):
    """Random Perturbations for L2 norm."""

    def get_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate random perturbation for the L2 norm.

        Parameters
        ----------
        x : torch.Tensor
            Input samples to perturb.

        Returns
        -------
        torch.Tensor
            Perturbed samples.
        """
        return torch.randn_like(x)

    @property
    def _constraint(self) -> type[L2Constraint]:
        return L2Constraint


class RandomPerturbL0(RandomPerturbBase):
    """Random Perturbations for L0 norm."""

    def get_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate random perturbation for the L0 norm.

        Parameters
        ----------
        x : torch.Tensor
            Input samples to perturb.

        Returns
        -------
        torch.Tensor
            Perturbed samples.
        """
        perturbations = torch.randn_like(x)
        return perturbations.sign()

    @property
    def _constraint(self) -> type[L0Constraint]:
        return L0Constraint


class RandomPerturb:
    """Random perturbation creator."""

    def __new__(cls, p: str, epsilon: float) -> RandomPerturbBase:
        """
        Creator for random perturbation in Lp norms.

        Parameters
        ----------
        p : str
            p-norm used for the random perturbation shape.
        epsilon : float
            Radius of the random perturbation constraint.

        Returns
        -------
        RandomPerturbBase
            Random perturbation object.

        Raises
        ------
        ValueError
            Raises ValueError if the norm is not in 0, 1, 2, inf.
        """
        random_inits = {
            LpPerturbationModels.L0: RandomPerturbL0,
            LpPerturbationModels.L1: RandomPerturbL1,
            LpPerturbationModels.L2: RandomPerturbL2,
            LpPerturbationModels.LINF: RandomPerturbLinf,
        }
        selected = random_inits.get(p)
        if selected is not None:
            return selected(epsilon=epsilon)
        msg = "Perturbation model not available."
        raise ValueError(msg)
