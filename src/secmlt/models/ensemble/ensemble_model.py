"""Model for ensembling multiple models."""
import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.ensemble.ensemble_function import (
    BaseEnsembleFunction,
    RawEnsembleFunction,
)
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from torch.utils.data import DataLoader


class EnsembleModel(BaseModel):
    """Generic class for ensemble of models."""

    def __init__(
            self,
            models: dict[str, BasePytorchClassifier],
            ensemble_function: BaseEnsembleFunction = None,
    ) -> None:
        """
        Create the ensemble model.

        Parameters
        ----------
        models : dict[str, BasePytorchClassifier]
            A dict containing as values the ensemble models, wrapped into a
            BasePytorchClassifier, and an identifier name as key.
            The input and output shapes of the models must be the same.
            If needed, define preprocessing and postprocessing for each model.
            Models are expected to return logits and not probabilities scores.
        ensemble_function : BaseEnsembleFunction, optional
            The function to use for aggregating the outputs of the models.
            Default: RawEnsembleFunction, which does not aggregate the outputs
            but returns them as a tensor of shape (n_models, batch_size, *model_output).
        """
        self.models = models
        self.ensemble_function = ensemble_function
        if not ensemble_function:
            self.ensemble_function = RawEnsembleFunction()
        super().__init__()

    def _get_device(self) -> torch.device:
        """
        Get the default device for the ensemble model.

        Use the device of the first model in the ensemble.

        Returns
        -------
        torch.device
            The device of the first model in the ensemble.
        """
        return next(iter(self.models.values()))._get_device()

    def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the decision function of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Output scores from the model.
        """
        x = x.to(device=self._get_device())
        return self.ensemble_function(x, self.models)

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Prediction function for the ensemble model.

        Return the predicted class for the given samples. If the ensembling
        strategy is RawEnsembleFunction, use majority voting among models
        predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Predicted class for the samples.
        """
        scores = self.decision_function(x)
        predictions = torch.argmax(scores, dim=-1)
        return predictions.mode(dim=0).values if isinstance(
            self.ensemble_function, RawEnsembleFunction) else predictions

    def gradient(self, x: torch.Tensor, y: int, *args, **kwargs) -> torch.Tensor:
        """
        Compute gradients of the score y w.r.t. x.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.
        y : int
            Target score.

        Returns
        -------
        torch.Tensor
            Input gradients of the target score y.
        """
        x = x.detach().clone().requires_grad_().to(self._get_device())
        if x.grad is not None:
            x.grad.zero_()
        output = self.decision_function(x)
        output = output[..., y].sum()
        output.backward()
        return x.grad

    def train(self, dataloader: DataLoader) -> "BaseModel":
        """
        Train the model with the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Train data loader.
        """
        raise NotImplementedError
