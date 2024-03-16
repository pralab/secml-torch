"""Interface for the data processing functionalities."""

from abc import ABC, abstractmethod

import torch


class DataProcessing(ABC):
    """Abstract data processing class."""

    @abstractmethod
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverted transform (if defined).

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            The samples in the input space before the transformation.
        """
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            The samples after transformation.
        """
        return self._process(x)
