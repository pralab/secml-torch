import tensorflow as tf
import torch

from src.models.base_model import BaseModel
from src.models.preprocessing.preprocessing import Preprocessing
from src.models.tensorflow.base_tensorflow_trainer import BaseTensorflowTrainer


class BaseTensorflowClassifier(BaseModel):
    def __init__(
        self,
        model: tf.keras.Model,
        preprocessing: Preprocessing = None,
        trainer: BaseTensorflowTrainer = None,
    ):
        super().__init__(preprocessing=preprocessing)
        self._model = model
        self._trainer = trainer

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO
        Parameters
        ----------
        x :

        Returns
        -------

        """
        pass

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO
        Parameters
        ----------
        x :

        Returns
        -------

        """
        pass

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        TODO
        Parameters
        ----------
        x :

        Returns
        -------

        """
        pass

    def train(self, dataloader: torch.Tensor):
        """
        TODO
        Parameters
        ----------
        dataloader :

        Returns
        -------

        """
        pass
