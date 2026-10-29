"""Ensemble function that averages model outputs."""
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from .raw import RawEnsembleFunction


class AvgEnsembleFunction(RawEnsembleFunction):
    """
    Average ensemble function.

    Implements a simple ensemble function that averages the outputs of the
    ensemble models. The gradient is computed accordingly.
    """

    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier]
    ) -> torch.Tensor:
        """
        Average ensemble forward function.

        Forward the input through all the models in the ensemble, and returns
        the averaged output.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor
        models : dict[str, BasePytorchClassifier]
            The ensemble model

        Returns
        -------
        torch.Tensor
            The ensemble output
        """
        outputs = super().forward(input, models)
        return outputs.mean(dim=0)
