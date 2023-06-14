from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.models.data_processing.identity_data_processing import IdentityDataProcessing
from src.models.data_processing.data_processing import DataProcessing


class BaseModel(ABC):
    def __init__(
        self,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
    ):
        """
        Create base abstract model
        Parameters
        ----------
        preprocessing : DataProcessing
        postprocessing: DataProcessing

        """
        self._preprocessing = (
            preprocessing if preprocessing is not None else IdentityDataProcessing()
        )
        self._postprocessing = (
            postprocessing if postprocessing is not None else IdentityDataProcessing()
        )

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocessing(x)
        x = self._decision_function(x)
        x = self._postprocessing(x)
        return x

    @abstractmethod
    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        ...

    @abstractmethod
    def train(self, dataloader: DataLoader):
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.decision_function(x)
