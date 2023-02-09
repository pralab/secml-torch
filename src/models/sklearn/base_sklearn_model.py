from typing import Callable

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

from src.data.sklearn_dataset import SklearnDataset
from src.models.base_model import BaseModel
from src.models.preprocessing.preprocessing import Preprocessing


class BaseSklearnModel(BaseModel):
    def __init__(
        self, model: sklearn.base.BaseEstimator, preprocessing: Preprocessing = None
    ):
        super().__init__(preprocessing=preprocessing)
        self._model: sklearn.base.BaseEstimator = model

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self._model, "decision_function"):
            return self.to_tensor(self._model.decision_function(self.to_2d_numpy(x)))
        elif hasattr(self._model, "predict_proba"):
            return self.to_tensor(self._model.predict_proba(self.to_2d_numpy(x)))
        raise AttributeError(
            "This model has neither decision_function nor predict_proba."
        )

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        raise NotImplementedError(
            "Custom sklearn model do not implement gradients. "
            "Use specific class or create subclass with custom definition."
        )

    def train(self, dataloader: DataLoader):
        if not isinstance(dataloader.dataset, SklearnDataset):
            raise ValueError(
                f"Internal dataset is not SklearnDataset, but {type(dataloader.dataset)}"
            )
        x, y = dataloader.dataset.x, dataloader.dataset.y
        self._model.fit(x, y)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x_numpy = self.to_2d_numpy(x)
        y = self._model.predict(x_numpy)
        y = self.to_tensor(y)
        return y

    @classmethod
    def to_numpy(cls, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    @classmethod
    def to_tensor(cls, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x)

    @classmethod
    def to_2d_numpy(cls, x: torch.Tensor) -> np.ndarray:
        return x.view(x.shape[0], -1).cpu().detach().numpy()
