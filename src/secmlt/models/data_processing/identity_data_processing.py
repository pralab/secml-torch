"""Identity data processing, returns the samples as they are."""

import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class IdentityDataProcessing(DataProcessing):
    """Identity transformation."""

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identity transformation. Returns the samples unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Unchanged samples.
        """
        return x

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the sample as it is.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Unchanged samples for identity inverse transformation.
        """
        return x
