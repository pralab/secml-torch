"""Mean Standard normalization data processing."""

from collections.abc import Sequence

import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class MeanStdNormalization(DataProcessing):
    """Normalizes input samples with fixed mean and standard deviation."""

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        """
        Create input normalization.

        Parameters
        ----------
        mean : Sequence[float]
            Per-channel mean values.
        std : Sequence[float]
            Per-channel standard deviation values.
        """
        self._mean = torch.tensor(mean)
        self._std = torch.tensor(std)

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input samples.

        Parameters
        ----------
        x : torch.Tensor
            Input samples with shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Normalized samples.
        """
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        return (x - mean[None, :, None, None]) / std[None, :, None, None]

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize samples.

        Parameters
        ----------
        x : torch.Tensor
            Normalized samples with shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Denormalized samples.
        """
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        return x * std[None, :, None, None] + mean[None, :, None, None]
