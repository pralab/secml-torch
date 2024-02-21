from abc import abstractmethod
from typing import Callable, List, Union

import torch
from torch.utils.data import DataLoader, TensorDataset

from secml2.adv.evasion.perturbation_models import PerturbationModels
from secml2.adv.backends import Backends
from secml2.models.base_model import BaseModel
from secml2.trackers.tracker import Tracker


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

    @classmethod
    def get_foolbox_implementation(cls):
        try:
            import foolbox
        except ImportError:
            raise ImportError("Foolbox extra not installed.")
        else:
            return cls._get_foolbox_implementation()

    @staticmethod
    def _get_foolbox_implementation():
        raise NotImplementedError("Foolbox implementation not available.")

    @staticmethod
    def get_native_implementation():
        raise NotImplementedError("Native implementation not available.")


class BaseEvasionAttack:
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
        adversarials = []
        original_labels = []
        for samples, labels in data_loader:
            x_adv = self._run(model, samples, labels)
            adversarials.append(x_adv)
            original_labels.append(labels)
        adversarials = torch.vstack(adversarials)
        original_labels = torch.hstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        adversarial_loader = DataLoader(
            adversarial_dataset, batch_size=data_loader.batch_size
        )
        return adversarial_loader

    @property
    def trackers(self) -> Union[List[Tracker], None]:
        return self._trackers

    @trackers.setter
    def trackers(self, trackers: Union[List[Tracker], None]) -> None:
        if trackers is not None:
            raise NotImplementedError("Trackers are not implemented for this backend")

    @abstractmethod
    def _run(self, model: BaseModel, samples: torch.Tensor, labels: torch.Tensor):
        """
        Compute the attack against the model, using the input data (batch).
        It returns the batch of adversarial examples.
        """
        ...
