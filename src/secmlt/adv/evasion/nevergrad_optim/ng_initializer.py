"""Initializer for the manipulation in Nevergrad."""
import nevergrad
import torch
from secmlt.optimization.initializer import Initializer


class NevergradInitializer(Initializer):
    """Initialize manipulation by wrapping the requirements of nevergrad."""

    def __init__(self, initializer: Initializer, lb: float = 0, ub: float = 0) -> None:
        """
        Create the manipulation to use in nevergrad.

        Parameters
        ----------
        initializer : Initializer
            the initialization to apply
        lb : float = 0.0
            lower bound for initialization
        ub : float = 1
            upper bound for initialization
        """
        self.initializer = initializer
        self.ub = ub
        self.lb = lb
        super().__init__()

    def __call__(self, x: torch.Tensor, **kwargs) -> nevergrad.p.Array:
        """
        Create the initialization.

        Parameters
        ----------
        x : torch.Tensor
            the sample from which compute the manipulation.

        Returns
        -------
        nevergrad.p.Array
            the initialized delta.
        """
        init_delta = self.initializer(x)
        delta = nevergrad.p.Array(shape=init_delta.shape, lower=-self.ub, upper=self.ub)
        delta.value = init_delta.numpy()
        return delta
