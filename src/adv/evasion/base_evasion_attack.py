from abc import abstractmethod
from typing import Callable

from torch.utils.data import DataLoader

from src.adv.backends import Backends
from src.adv.evasion.perturbation_models import PerturbationModels
from src.models.base_model import BaseModel


class BaseEvasionAttackCreator:
    @classmethod
    def get_implementation(cls, backend: str) -> Callable:
        implementations = {
            Backends.FOOLBOX: cls.get_foolbox_implementation,
            Backends.NATIVE: cls.get_native_implementation,
        }
        if backend not in implementations:
            raise NotImplementedError("Unsupported or not-implemented backend.")
        return implementations[backend]()

    @staticmethod
    def check_perturbation_model_available(perturbation_model: str) -> bool:
        if not PerturbationModels.is_perturbation_model_available(perturbation_model):
            raise NotImplementedError("Unsupported or not-implemented threat model.")

    @staticmethod
    def get_foolbox_implementation():
        raise NotImplementedError("Foolbox implementation not available.")

    @staticmethod
    def get_native_implementation():
        raise NotImplementedError("Native implementation not available.")


class BaseEvasionAttack:
    @abstractmethod
    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        """
        Compute the attack against the model, using the input data.
        It returns a dataset with the adversarial examples and the original labels
        :param model: model to test
        :type model: BaseModel
        :param data_loader: input data
        :type data_loader: DataLoader
        :return: Data loader with adversarial examples and original labels
        :rtype: DataLoader
        """
        ...
