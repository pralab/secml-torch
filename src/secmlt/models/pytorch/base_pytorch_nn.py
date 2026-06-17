"""Wrappers for PyTorch models."""

from collections.abc import Callable

import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.utils.data import DataLoader


class BasePyTorchClassifier(BaseModel):
    """Wrapper for PyTorch classifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: Callable | None = None,
        postprocessing: Callable | None = None,
        trainer: BasePyTorchTrainer = None,
    ) -> None:
        """
        Create wrapped PyTorch classifier.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model.
        preprocessing : callable, optional
            Callable applied to inputs before the forward pass, by default None.
            Accepts any callable: a ``DataProcessing`` subclass,
            a ``torchvision.transforms`` transform, or any ``torch.nn.Module``.
        postprocessing : callable, optional
            Callable applied to outputs after the forward pass, by default None.
        trainer : BasePyTorchTrainer, optional
            Trainer object to train the model, by default None.
        """
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing)
        self._model: torch.nn.Module = model
        self._trainer = trainer

    @property
    def model(self) -> torch.nn.Module:
        """
        Get the wrapped instance of PyTorch model.

        Returns
        -------
        torch.nn.Module
            Wrapped PyTorch model.
        """
        return self._model

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the predicted class for the given samples.

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
        return torch.argmax(scores, dim=-1)

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute decision function of the model.

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
        return self._model(x)

    def train(self, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train the model with given dataloader, if trainer is set.

        Parameters
        ----------
        dataloader : DataLoader
            Training PyTorch dataloader to use for training.

        Returns
        -------
        torch.nn.Module
            Trained PyTorch model.

        Raises
        ------
        ValueError
            Raises ValueError if the trainer is not set.
        """
        if self._trainer is None:
            msg = "Cannot train without a trainer."
            raise ValueError(msg)
        return self._trainer.train(self._model, dataloader)
