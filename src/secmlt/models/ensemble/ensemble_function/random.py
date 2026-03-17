from .base import BaseEnsembleFunction
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


class RandomEnsembleFunction(BaseEnsembleFunction):
    """
    Implements a random ensemble function where each forward considers one
    randomly picked model. The gradient is computed accordingly.
    """

    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier]
    ) -> torch.Tensor:
        """
        Forwards the input through a randomly picked model.

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
        input = input.to(models[model_name]._get_device())
        output = models[model_name](input).to(
            list(models.values())[0]._get_device())
        return self._apply_scaling(output, model_name)
