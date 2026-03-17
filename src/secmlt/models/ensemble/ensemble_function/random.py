"""Ensemble function that randomly selects a model for each forward pass."""
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from .base import BaseEnsembleFunction


class RandomEnsembleFunction(BaseEnsembleFunction):
    """
    Random ensemble function.

    Implements a random ensemble function where each forward considers one
    randomly selected model. The gradient is computed accordingly.
    """

    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier]
    ) -> torch.Tensor:
        """
        Forward the input through a randomly picked model.

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
        i = torch.randint(0, len(models), (1,)).item()
        model_name = list(models.keys())[i]
        _input = input.to(models[model_name]._get_device())
        output = models[model_name](_input).to(
            next(iter(models.values()))._get_device()
        )
        return self._apply_scaling(output, model_name)
