"""Model trainers."""

from abc import ABCMeta, abstractmethod

from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader


class BaseAdvTrainer(metaclass=ABCMeta):
    """Abstract class for model trainers."""

    @abstractmethod
    def train(self, model: BaseModel,
              dataloader: DataLoader,
              attack: BaseEvasionAttack) -> BaseModel:
        """
        Train a model with the given dataloader.

        Parameters
        ----------
        model : BaseModel
            Model to train.
        dataloader : DataLoader
            Training dataloader.
        attack : BaseAttack
            Adversarial attack to use for adversarial training.

        Returns
        -------
        BaseModel
            The trained model.
        """
        ...
