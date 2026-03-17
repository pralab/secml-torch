from .base import BaseEnsembleFunction
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


class RawEnsembleFunction(BaseEnsembleFunction):
    """
    Performs forward and backward pass on each ensemble model separately, and
    returns the unaggregated results in a single tensor.
    """

    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier],
    ) -> torch.Tensor:
        """
        Forwards the input through all the models in the ensemble, and returns
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
        outputs = torch.stack([
            self._apply_scaling(
                model(input.to(model._get_device())).to(
                    list(models.values())[0]._get_device()),
                model_name) for model_name, model in models.items()]
        )
        return outputs
