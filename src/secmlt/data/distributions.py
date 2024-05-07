"""Implementation for uncommon distributions."""

from abc import ABC, abstractmethod

import torch
from torch.distributions.gamma import Gamma


class Distribution(ABC):
    """Abstract class for distributions."""

    @abstractmethod
    def sample(self, shape: torch.Size) -> torch.Tensor:
        """
        Sample from the distribution.

        This method generates a sample from the distribution, with the specified shape.
        If no shape is specified, a single sample is returned.

        Parameters
        ----------
        shape : torch.Size, optional
            The shape of the sample to be generated. Default is torch.Size(), which
            corresponds to a single sample.

        Returns
        -------
        torch.Tensor
            A tensor of samples from the distribution, with the specified shape.
        """
        ...


class Rademacher(Distribution):
    """Samples from Rademacher distribution (-1, 1) with equal probability."""

    def sample(self, shape: torch.Size) -> torch.Tensor:
        """
        Sample from the Rademacher distribution.

        This method generates a sample from the Rademacher distribution, where each
        sample is either -1 or 1 with equal probability. The shape of the output
        is determined by the `shape` parameter.

        Parameters
        ----------
        shape : torch.Size
            The shape of the sample to be generated.

        Returns
        -------
        torch.Tensor
            A tensor of samples from the Rademacher distribution, with values -1 or 1.

        Examples
        --------
        >>> dist = Rademacher()
        >>> sample = dist.sample((3, 4))
        """
        _prob = 0.5
        return torch.where((torch.rand(size=shape) < _prob), -1, 1)


class GeneralizedNormal(Distribution):
    r"""
    Generalized normal distribution.

    .. math::
        f(x; \mu, \alpha, \beta) = \frac{\beta}{2 \alpha
        \Gamma(1 / \beta)} e^{-(|x-\mu| / \alpha)^\beta}

    where `\mu` is the location parameter, `\alpha` is the scale
    parameter, and `\beta` is the shape parameter.
    """

    def sample(self, shape: torch.Size, p: float = 2) -> torch.Tensor:
        """
        Sample from the generalized normal distribution.

        This method generates a sample from the generalized normal
        distribution, with shape parameter `p`. The shape of the
        output is determined by the `shape` parameter.

        Parameters
        ----------
        shape : torch.Size
            The shape of the sample to be generated.
        p : float, optional
            The shape parameter of the generalized normal distribution. Default is 2.

        Returns
        -------
        torch.Tensor
            A tensor of samples from the generalized normal distribution.

        Examples
        --------
        >>> dist = GeneralizedNormal()
        >>> sample = dist.sample((3, 4))
        """
        g = Gamma(concentration=1 / p, rate=1).sample(sample_shape=shape)
        r = Rademacher().sample(shape=shape)
        return r * g ** (1 / p)
