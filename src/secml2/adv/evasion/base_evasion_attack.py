from abc import abstractmethod
from typing import Callable, List, Type, Union
from secml2.adv.evasion.perturbation_models import PerturbationModels

import torch
from torch.utils.data import DataLoader, TensorDataset

from secml2.adv.backends import Backends
from secml2.models.base_model import BaseModel

# lazy evaluation to avoid circular imports
TRACKER_TYPE = "secml2.trackers.tracker.Tracker"


class BaseEvasionAttackCreator:
    @classmethod
    def get_implementation(cls, backend: str) -> Callable:
        implementations = {
            Backends.FOOLBOX: cls.get_foolbox_implementation,
            Backends.NATIVE: cls.get_native_implementation,
        }
        cls.check_backend_available(backend)
        return implementations[backend]()

    @classmethod
    def check_backend_available(cls, backend: str) -> bool:
        if backend in cls.get_backends():
            return True
        raise NotImplementedError("Unsupported or not-implemented backend.")

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

    @staticmethod
    @abstractmethod
    def get_backends():
        raise NotImplementedError("Backends should be specified in inherited class.")


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
    def trackers(self) -> Union[List[Type[TRACKER_TYPE]], None]:
        return self._trackers

    @trackers.setter
    def trackers(self, trackers: Union[List[Type[TRACKER_TYPE]], None] = None) -> None:
        if self.trackers_allowed():
            if not isinstance(trackers, list):
                trackers = [trackers]
            self._trackers = trackers
        elif trackers is not None:
            raise NotImplementedError("Trackers not implemented for this attack.")

    @abstractmethod
    def trackers_allowed(cls):
        return False

    @classmethod
    def check_perturbation_model_available(cls, perturbation_model: str) -> bool:
        if perturbation_model in cls.get_perturbation_models():
            return
        raise NotImplementedError("Unsupported or not-implemented perturbation model.")

    @staticmethod
    @abstractmethod
    def get_perturbation_models():
        raise NotImplementedError(
            "Perturbation models should be specified in inherited class."
        )

    @abstractmethod
    def _run(self, model: BaseModel, samples: torch.Tensor, labels: torch.Tensor):
        """
        Compute the attack against the model, using the input data (batch).
        It returns the batch of adversarial examples.
        """
        ...
