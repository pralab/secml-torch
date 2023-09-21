from typing import Callable, Optional

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

from secml2.data.sklearn_dataloader import SklearnDataLoader
from secml2.models.base_model import BaseModel
from secml2.models.data_processing.data_processing import DataProcessing
from secml2.models.sklearn.sklearn_layer import SklearnLayer, as_array, as_tensor


class BaseSklearnModel(BaseModel):
    def __init__(
        self, model: sklearn.base.BaseEstimator, preprocessing: DataProcessing = None
    ):
        super().__init__(preprocessing=preprocessing)
        self._clf = model
        self._model = SklearnLayer(model)

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self._clf, "decision_function"):
            return self.to_tensor(self._clf.decision_function(self.to_2d_numpy(x)))
        elif hasattr(self._clf, "predict_proba"):
            return self.to_tensor(self._clf.predict_proba(self.to_2d_numpy(x)))
        raise AttributeError(
            "This model has neither decision_function nor predict_proba."
        )

    def _gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        raise NotImplementedError(
            "Custom sklearn model do not implement gradients. "
            "Use specific class or create subclass with custom definition."
        )
    
    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        return self.to_tensor(self._gradient(x))

    def train(self, dataloader: DataLoader):
        if not isinstance(dataloader.dataset, SklearnDataLoader):
            dataloader = SklearnDataLoader(dataloader)
        x, y = dataloader.dataset.x, dataloader.dataset.y
        self._clf.fit(x, y)
        return self

    @classmethod
    def to_numpy(cls, x: torch.Tensor) -> np.ndarray:
        return as_array(x)

    @classmethod
    def to_tensor(cls, x: np.ndarray) -> torch.Tensor:
        return as_tensor(x)

    @classmethod
    def to_2d_numpy(cls, x: torch.Tensor) -> np.ndarray:
        return as_array(x.view(x.shape[0], -1))
