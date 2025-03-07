"""Processing functions for gradients."""

from abc import ABC, abstractmethod

import torch.linalg
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from torch.nn.functional import normalize


def lin_proj_l1(x: torch.Tensor) -> torch.Tensor:
    """Return the linear projection of x onto an L1 unit ball.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to project.

    Returns
    -------
    torch.Tensor
    Linear projection of x onto unit L1 ball.
    """
    w = abs(x)
    num_max = (w == w.max()).sum()
    w = torch.where(w == w.max(), 1 / num_max, 0)
    return w * x.sign()


class GradientProcessing(ABC):
    """Gradient processing base class."""

    @abstractmethod
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Process the gradient with the given transformation.

        Parameters
        ----------
        grad : torch.Tensor
            Input gradients.

        Returns
        -------
        torch.Tensor
            The processed gradients.
        """
        ...


class NoGradientProcessing(GradientProcessing):
    """Absence of processing."""

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Gradient is not processed.

        Parameters
        ----------
        grad : torch.Tensor
            Input gradients.

        Returns
        -------
        torch.Tensor
            The unaltered gradient.
        """
        return grad


class LinearProjectionGradientProcessing(GradientProcessing):
    """Linear projection of the gradient onto Lp balls."""

    def __init__(self, perturbation_model: str = LpPerturbationModels.L2) -> None:
        """
        Create linear projection for the gradient.

        Parameters
        ----------
        perturbation_model : str, optional
            Perturbation model for the Lp ball, by default LpPerturbationModels.L2.

        Raises
        ------
        ValueError
            Raises ValueError if the perturbation model is not implemented.
            Available, l1, l2, linf.
        """
        perturbations_models = {
            LpPerturbationModels.L1: 1,
            LpPerturbationModels.L2: 2,
            LpPerturbationModels.LINF: float("inf"),
        }
        if perturbation_model not in perturbations_models:
            msg = f"{perturbation_model} not available. \
                Use one of: {perturbations_models.values()}"
            raise ValueError(msg)
        self.p = perturbations_models[perturbation_model]

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Process gradient with linear projection onto the Lp ball.

        Sets the direction by maximizing the scalar product with the
        gradient over the Lp ball.

        Parameters
        ----------
        grad : torch.Tensor
            Input gradients.

        Returns
        -------
        torch.Tensor
            The gradient linearly projected onto the Lp ball.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if the norm is not in 2, inf.
        """
        original_shape = grad.data.shape
        if self.p == LpPerturbationModels.get_p(LpPerturbationModels.L2):
            return normalize(grad.data.flatten(start_dim=1), p=self.p, dim=1).view(
                original_shape
            )
        if self.p == LpPerturbationModels.get_p(LpPerturbationModels.L1):
            return lin_proj_l1(grad.data.flatten(start_dim=1)).view(original_shape)
        if self.p == LpPerturbationModels.get_p(LpPerturbationModels.LINF):
            return torch.sign(grad)
        msg = "Only L2 and LInf norms implemented now"
        raise NotImplementedError(msg)
