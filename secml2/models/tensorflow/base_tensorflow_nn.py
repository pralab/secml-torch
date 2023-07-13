import tensorflow as tf
import torch

from secml2.models.base_model import BaseModel
from secml2.models.data_processing.data_processing import DataProcessing
from secml2.models.tensorflow.base_tensorflow_trainer import BaseTensorflowTrainer


class BaseTensorflowClassifier(BaseModel):
    def __init__(
        self,
        model: tf.keras.Model,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTensorflowTrainer = None,
    ):
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing)
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

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
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
