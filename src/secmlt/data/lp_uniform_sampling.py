"""Implementation of Lp uniform sampling."""

import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.data.distributions import GeneralizedNormal
from torch.distributions.exponential import Exponential


class LpUniformSampling:
    """
    Uniform sampling from the unit Lp ball.

    This class provides a method for sampling uniformly from the
    unit Lp ball, where Lp is a norm defined by a parameter `p`.
    The class supports sampling from the L0, L2, and Linf norms.

    The sampling method is based on the following reference:
    https://arxiv.org/abs/math/0503650

    Attributes
    ----------
    p : str
        The norm to use for sampling. Must be one of 'l0', 'l1', 'l2', 'linf'.
    """

    def __init__(self, p: str = LpPerturbationModels.L2) -> None:
        """
        Initialize the LpUniformSampling object.

        Parameters
        ----------
        p : str, optional
            The norm to use for sampling. Must be one
            of 'L0', 'L2', or 'Linf'. Default is 'L2'.
        """
        self.p = p

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the unit Lp ball with the same shape as a given tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor whose shape is used to determine the shape of the samples.

        Returns
        -------
        torch.Tensor
            A tensor of samples from the unit Lp ball, with the
            same shape as the input tensor `x`.
        """
        num_samples, dim = x.flatten(1).shape
        return self.sample(num_samples, dim).reshape(x.shape)

    def sample(self, num_samples: int = 1, dim: int = 2) -> torch.Tensor:
        """
        Sample uniformly from the unit Lp ball.

        This method generates a specified number of samples
        from the unit Lp ball, where Lp is a norm defined by the `p` parameter.
        The samples are generated using the algorithm
        described in the class documentation.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        dim : int
            The dimension of the samples.

        Returns
        -------
        torch.Tensor
            A tensor of samples from the unit Lp ball, with shape `(num_samples, dim)`.
        """
        shape = torch.Size((num_samples, dim))
        _p = LpPerturbationModels.get_p(self.p)

        if self.p == LpPerturbationModels.LINF:
            ball = 2 * torch.rand(size=shape) - 1
        elif self.p == LpPerturbationModels.L0:
            ball = torch.rand(size=shape).sign()
        else:
            g = GeneralizedNormal().sample(shape)
            e = Exponential(rate=1).sample(sample_shape=(num_samples,))
            d = ((torch.abs(g) ** _p).sum(-1) + e) ** (1 / _p)
            ball = g / d.unsqueeze(-1)

        return ball
