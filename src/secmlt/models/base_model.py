"""Basic wrapper for generic model."""

from abc import ABC, abstractmethod

import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.models.data_processing.identity_data_processing import (
    IdentityDataProcessing,
)
from torch.utils.data import DataLoader


class BaseModel(ABC):
    """Basic model wrapper."""

    def __init__(
        self,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
    ) -> None:
        """
        Create base model.

        Parameters
        ----------
        preprocessing : DataProcessing, optional
            Preprocessing to apply before the forward, by default None.
        postprocessing : DataProcessing, optional
            Postprocessing to apply after the forward, by default None.
        """
        self._preprocessing = (
            preprocessing if preprocessing is not None else IdentityDataProcessing()
        )
        self._postprocessing = (
            postprocessing if postprocessing is not None else IdentityDataProcessing()
        )

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return output predictions for given model.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Predictions from the model.
        """
        ...

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the decision function from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input damples.

        Returns
        -------
        torch.Tensor
            Model output scores.
        """
        x = self._preprocessing(x)
        x = self._decision_function(x)
        return self._postprocessing(x)

    @abstractmethod
    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Specific decision function of the model (data already preprocessed).

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input samples.

        Returns
        -------
        torch.Tensor
            Model output scores.
        """
        ...

    @abstractmethod
    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
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
        ...

    @abstractmethod
    def train(self, dataloader: DataLoader) -> "BaseModel":
        """
        Train the model with the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Train data loader.
        """
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Model ouptut scores.
        """
        return self.decision_function(x)
