"""Fixtures used for testing."""

import pytest
import torch
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.tests.mocks import MockModel
from secmlt.trackers.trackers import (
    GradientNormTracker,
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
    ScoresTracker,
)
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
def data() -> torch.Tensor:
    """
    Get random samples.

    Returns
    -------
    torch.Tensor
        A fake tensor with samples.
    """
    return torch.randn(10, 3, 32, 32).clamp(0.0, 1.0)


@pytest.fixture()
def labels() -> torch.Tensor:
    """
    Get random labels.

    Returns
    -------
    torch.Tensor
        A fake tensor with labels.
    """
    return torch.randint(0, 9, 10)


@pytest.fixture()
def loss_values() -> torch.Tensor:
    """
    Get random model outputs.

    Returns
    -------
    torch.Tensor
        A fake tensor with model outputs.
    """
    return torch.randn(10)


@pytest.fixture()
def output_values() -> torch.Tensor:
    """
    Get random model outputs.

    Returns
    -------
    torch.Tensor
        A fake tensor with model outputs.
    """
    return torch.randn(10, 10)


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


@pytest.fixture()
def loss_tracker() -> LossTracker:
    """
    Get loss tracker.

    Returns
    -------
    LossTracker
        A tracker for the loss.
    """
    return LossTracker()


@pytest.fixture()
def scores_tracker() -> ScoresTracker:
    """
    Get scores tracker.

    Returns
    -------
    LossTracker
        A tracker for the scores.
    """
    return ScoresTracker()


@pytest.fixture()
def prediction_tracker() -> PredictionTracker:
    """
    Get prediction tracker.

    Returns
    -------
    LossTracker
        A tracker for the predictions.
    """
    return PredictionTracker()


@pytest.fixture()
def perturbation_norm_tracker() -> PerturbationNormTracker:
    """
    Get perturbation norm tracker.

    Returns
    -------
    LossTracker
        A tracker for the perturbation norms.
    """
    return PerturbationNormTracker()


@pytest.fixture()
def gradient_norm_tracker() -> GradientNormTracker:
    """
    Get gradient norm tracker.

    Returns
    -------
    LossTracker
        A tracker for the gradient norms.
    """
    return GradientNormTracker()
