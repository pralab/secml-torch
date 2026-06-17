"""Basic wrapper for generic model."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from secmlt.models.data_processing.identity_data_processing import (
    IdentityDataProcessing,
)
from torch.utils.data import DataLoader


class BaseModel(ABC):
    """Basic model wrapper."""

    def __init__(
        self,
        preprocessing: Callable | None = None,
        postprocessing: Callable | None = None,
    ) -> None:
        """
        Create base model.

        Parameters
        ----------
        preprocessing : callable, optional
            Callable applied to inputs before the forward pass, by default None.
            Accepts any callable: a ``DataProcessing`` subclass,
            a ``torchvision.transforms`` transform, or any ``torch.nn.Module``.
        postprocessing : callable, optional
            Callable applied to outputs after the forward pass, by default None.
        """
        self._preprocessing = (
            preprocessing if preprocessing is not None else IdentityDataProcessing()
        )
        self._postprocessing = (
            postprocessing if postprocessing is not None else IdentityDataProcessing()
        )

    @abstractmethod
    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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

    def decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Return the decision function from the model.

        Requires override to specify custom args and kwargs passing.

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
    def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
    def train(self, dataloader: DataLoader) -> "BaseModel":
        """
        Train the model with the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Train data loader.
        """
        ...

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
        return self.decision_function(x, *args, **kwargs)
