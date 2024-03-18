"""Base classes for implementing attacks and wrapping backends."""

import importlib
from abc import abstractmethod
from typing import Literal

import torch
from secmlt.adv.backends import Backends
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader, TensorDataset

# lazy evaluation to avoid circular imports
TRACKER_TYPE = "secmlt.trackers.tracker.Tracker"


class BaseEvasionAttackCreator:
    """Generic creator for attacks."""

    @classmethod
    def get_implementation(cls, backend: str) -> "BaseEvasionAttack":
        """
        Get the implementation of the attack with the given backend.

        Parameters
        ----------
        backend : str
            The backend for the attack. See secmlt.adv.backends for
            available backends.

        Returns
        -------
        BaseEvasionAttack
            Attack implementation.
        """
        implementations = {
            Backends.FOOLBOX: cls.get_foolbox_implementation,
            Backends.NATIVE: cls._get_native_implementation,
        }
        cls.check_backend_available(backend)
        return implementations[backend]()

    @classmethod
    def check_backend_available(cls, backend: str) -> bool:
        """
        Check if a given backend is available for the attack.

        Parameters
        ----------
        backend : str
            Backend string.

        Returns
        -------
        bool
            True if the given backend is implemented.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if the requested backend is not in
            the list of the possible backends (check secmlt.adv.backends).
        """
        if backend in cls.get_backends():
            return True
        msg = "Unsupported or not-implemented backend."
        raise NotImplementedError(msg)

    @classmethod
    def get_foolbox_implementation(cls) -> "BaseEvasionAttack":
        """
        Get the Foolbox implementation of the attack.

        Returns
        -------
        BaseEvasionAttack
            Foolbox implementation of the attack.

        Raises
        ------
        ImportError
            Raises ImportError if Foolbox extra is not installed.
        """
        if importlib.util.find_spec("foolbox", None) is not None:
            return cls._get_foolbox_implementation()
        msg = "Foolbox extra not installed."
        raise ImportError(msg)

    @staticmethod
    def _get_foolbox_implementation() -> "BaseEvasionAttack":
        msg = "Foolbox implementation not available."
        raise NotImplementedError(msg)

    @staticmethod
    def _get_native_implementation() -> "BaseEvasionAttack":
        msg = "Native implementation not available."
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def get_backends() -> set[str]:
        """
        Get the available backends for the given attack.

        Returns
        -------
        set[str]
            Set of implemented backends available for the attack.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if not implemented in the inherited class.
        """
        msg = "Backends should be specified in inherited class."
        raise NotImplementedError(msg)


class BaseEvasionAttack:
    """Base class for evasion attacks."""

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        """
        Compute the attack against the model, using the input data.

        Parameters
        ----------
        model : BaseModel
            Model to test.
        data_loader : DataLoader
            Test dataloader.

        Returns
        -------
        DataLoader
            Dataloader with adversarial examples and original labels.
        """
        adversarials = []
        original_labels = []
        for samples, labels in data_loader:
            x_adv, _ = self._run(model, samples, labels)
            adversarials.append(x_adv)
            original_labels.append(labels)
        adversarials = torch.vstack(adversarials)
        original_labels = torch.hstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        return DataLoader(
            adversarial_dataset,
            batch_size=data_loader.batch_size,
        )

    @property
    def trackers(self) -> list[TRACKER_TYPE] | None:
        """
        Get the trackers set for this attack.

        Returns
        -------
        list[TRACKER_TYPE] | None
            Trackers set for the attack, if any.
        """
        return self._trackers

    @trackers.setter
    def trackers(self, trackers: list[TRACKER_TYPE] | None = None) -> None:
        if self._trackers_allowed():
            if trackers is not None and not isinstance(trackers, list):
                trackers = [trackers]
            self._trackers = trackers
        elif trackers is not None:
            msg = "Trackers not implemented for this attack."
            raise NotImplementedError(msg)

    @classmethod
    @abstractmethod
    def _trackers_allowed(cls) -> Literal[False]:
        return False

    @classmethod
    def check_perturbation_model_available(cls, perturbation_model: str) -> bool:
        """
        Check whether the given perturbation model is available for the attack.

        Parameters
        ----------
        perturbation_model : str
            A perturbation model.

        Returns
        -------
        bool
            True if the attack implements the given perturbation model.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if not implemented in the inherited class.
        """
        if perturbation_model in cls.get_perturbation_models():
            return
        msg = "Unsupported or not-implemented perturbation model."
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def get_perturbation_models() -> set[str]:
        """
        Check the perturbation models implemented for the given attack.

        Returns
        -------
        set[str]
            The set of perturbation models for which the attack is implemented.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if not implemented in the inherited class.
        """
        msg = "Perturbation models should be specified in inherited class."
        raise NotImplementedError(msg)

    @abstractmethod
    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ...
