"""Ensemble function that returns raw model outputs."""
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from .base import BaseEnsembleFunction


class RawEnsembleFunction(BaseEnsembleFunction):
    """
    Raw ensemble function.

    Performs forward and backward pass on each ensemble model separately, and
    returns the unaggregated results in a single tensor.
    """

    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier],
    ) -> torch.Tensor:
        """
        Raw ensemble forward function.

        Forward the input through all the models in the ensemble, and returns
        the averaged output.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor
        models : dict[str, BasePytorchClassifier]
            The ensemble models

        Returns
        -------
        torch.Tensor
            The ensemble output
        """
        return torch.stack([
            self._apply_scaling(
                model(input.to(model._get_device())).to(
                    next(iter(models.values()))._get_device()),
                model_name) for model_name, model in models.items()]
        )
