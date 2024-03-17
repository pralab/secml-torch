"""Fixtures used for testing."""

import pytest
import torch
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.tests.mocks import MockModel
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture()
def data_loader() -> DataLoader[tuple[torch.Tensor]]:
    """
    Create fake data loader.

    Returns
    -------
    DataLoader[tuple[torch.Tensor]]
        A loader with random samples and labels.
    """
    # Create a dummy dataset loader for testing
    data = torch.randn(100, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture()
def adv_loaders() -> list[DataLoader[tuple[torch.Tensor, ...]]]:
    """
    Create fake adversarial loaders.

    Returns
    -------
    list[DataLoader[Tuple[torch.Tensor, ...]]]
        A list of multiple loaders (with same ordered labels).
    """
    # Create a list of dummy adversarial example loaders for testing
    loaders = []
    adv_labels = torch.randint(0, 10, (100,))
    for _ in range(3):
        adv_data = torch.randn(100, 3, 32, 32)
        adv_dataset = TensorDataset(adv_data, adv_labels)
        loaders.append(DataLoader(adv_dataset, batch_size=10))
    return loaders


@pytest.fixture()
def model() -> torch.nn.Module:
    """
    Create fake model.

    Returns
    -------
    torch.nn.Module
        Fake model.
    """
    return BasePytorchClassifier(model=MockModel())


@pytest.fixture()
def native_pgd_attack() -> BaseEvasionAttack:
    """Get native PGD."""
    return PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.5,
        num_steps=10,
        step_size=0.1,
        random_start=True,
        y_target=None,
        lb=0.0,
        ub=1.0,
        backend="native",
    )


@pytest.fixture()
def foolbox_pgd_attack() -> BaseEvasionAttack:
    """Get foolbox PGD."""
    return PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.5,
        num_steps=10,
        step_size=0.1,
        random_start=True,
        y_target=None,
        lb=0.0,
        ub=1.0,
        backend="foolbox",
    )
