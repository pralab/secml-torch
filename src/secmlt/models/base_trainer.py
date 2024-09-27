"""Model trainers."""

from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader

from secmlt.models.base_model import BaseModel


class BaseTrainer(metaclass=ABCMeta):
    """Abstract class for model trainers."""

    @abstractmethod
    def train(self, model: BaseModel, dataloader: DataLoader) -> BaseModel:
        """
        Train a model with the given dataloader.

        Parameters
        ----------
        model : BaseModel
            Model to train.
        dataloader : DataLoader
            Training dataloader.

        Returns
        -------
        BaseModel
            The trained model.
        """
        ...
