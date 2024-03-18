"""Initializers for the attacks."""

import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.optimization.random_perturb import RandomPerturb


class Initializer:
    """Initialization for the perturbation delta."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get initialization for the perturbation.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Initialized perturbation.
        """
        return torch.zeros_like(x)


class RandomLpInitializer(Initializer):
    """Random perturbation initialization in Lp ball."""

    def __init__(
        self,
        radius: torch.Tensor,
        perturbation_model: LpPerturbationModels,
    ) -> None:
        """
        Create random perturbation initializer.

        Parameters
        ----------
        radius : torch.Tensor
            Radius of the Lp ball for the constraint.
        perturbation_model : LpPerturbationModels
            Perturbation model for the constraint.
        """
        self.radius = radius
        self.perturbation_model = perturbation_model
        self.initializer = RandomPerturb(p=self.perturbation_model, epsilon=self.radius)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get random perturbations.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Initialized random perturbations.
        """
        return self.initializer(x)
