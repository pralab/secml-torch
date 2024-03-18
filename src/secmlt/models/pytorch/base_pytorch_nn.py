"""Wrappers for PyTorch models."""

import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.utils.data import DataLoader


class BasePytorchClassifier(BaseModel):
    """Wrapper for PyTorch classifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BasePyTorchTrainer = None,
    ) -> None:
        """
        Create wrapped PyTorch classifier.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model.
        preprocessing : DataProcessing, optional
            Preprocessing to apply before the forward., by default None.
        postprocessing : DataProcessing, optional
            Postprocessing to apply after the forward, by default None.
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

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        Compute batch gradients of class y w.r.t. x.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.
        y : int
            Class label.

        Returns
        -------
        torch.Tensor
            Gradient of class y w.r.t. input x.
        """
        x = x.clone().requires_grad_()
        if x.grad is not None:
            x.grad.zero_()
        output = self.decision_function(x)
        output = output[:, y].sum()
        output.backward()
        return x.grad

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
