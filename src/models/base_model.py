from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.models.preprocessing.identity_preprocessing import IdentityPreprocessing
from src.models.preprocessing.preprocessing import Preprocessing


class BaseModel(ABC):
    def __init__(self, preprocessing: Preprocessing = None):
        """
        Create base abstract model
        Parameters
        ----------
        preprocessing : Preprocessing

        """
        self._preprocessing = (
            preprocessing if preprocessing is not None else IdentityPreprocessing()
        )

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        ...

    @abstractmethod
    def train(self, dataloader: DataLoader):
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.decision_function(x)
