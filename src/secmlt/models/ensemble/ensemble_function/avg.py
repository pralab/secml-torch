from .raw import RawEnsembleFunction
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


class AvgEnsembleFunction(RawEnsembleFunction):
    """
    Implements a simple ensemble function that averages the outputs of the
    ensemble models. The gradient is computed accordingly.
    """

    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier]
    ) -> torch.Tensor:
        """
        Forwards the input through all the models in the ensemble, and returns
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
