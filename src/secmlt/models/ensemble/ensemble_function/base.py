from abc import ABC, abstractmethod
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BaseEnsembleFunction(ABC, torch.nn.Module):
    """Abstract class for ensemble functions."""

    def __init__(
            self,
            logits_scalers : dict[str, StandardScaler | MinMaxScaler] = None,
    ) -> None:
        """
        Creates the Ensemble function.

        Parameters
        ----------
        logits_scalers : dict[str, StandardScaler | MinMaxScaler], optional
            A dict containing as keys the identifiers of the models to which
            apply logits normalization, and already fitted scikit-learn scalers
            as values. It is not required to normalize the logits of all the
            ensemble models.
        """
        super().__init__()
        self._logits_scalers = logits_scalers

    @abstractmethod
    def forward(
            self,
            input: torch.Tensor,
            models: dict[str, BasePytorchClassifier],
    ) -> torch.Tensor:
        """
        Forwards the input through ensemble and returns the output based on the
        implemented strategy.

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
        raise NotImplementedError

    def _apply_scaling(
            self,
            logits: torch.Tensor,
            model_name: str
    ) -> torch.Tensor:
        """
        If a scaler is defined for the model, rescale the logits with it.

        Parameters
        ----------
        logits : torch.Tensor
            The logits tensor
        model_name :
            The identifier name of the model
        Returns
        -------
        torch.Tensor
            The scaled logits
        """
        if self._logits_scalers and model_name in self._logits_scalers:
            mean = torch.tensor(
                self._logits_scalers[model_name].mean_, device=logits.device)
            std = torch.tensor(
                self._logits_scalers[model_name].scale_, device=logits.device)
            return (logits - mean) / std
        else:
            return logits
