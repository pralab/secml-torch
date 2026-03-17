"""Fixtures used for testing."""

from unittest.mock import patch

import pytest
import torch
from secmlt.models.hugging_face.base_hf_lm import HFCausalLM
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.tests.mocks import MockModel
from secmlt.tests.mocks_lm import MockHFModel, MockHFTokenizer
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def dataset() -> TensorDataset:
    """Create fake dataset."""
    data = torch.randn(100, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 9, (100,))
    return TensorDataset(data, labels)


@pytest.fixture
def data_loader(dataset: TensorDataset) -> DataLoader[tuple[torch.Tensor]]:
    """
    Create fake data loader.

    Parameters
    ----------
    dataset : TensorDataset
        Dataset to wrap in the loader

    Returns
    -------
    DataLoader[tuple[torch.Tensor]]
        A loader with random samples and labels.

    """
    # Create a dummy dataset loader for testing
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
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
    adv_labels = torch.randint(0, 9, (100,))
    for _ in range(3):
        adv_data = torch.randn(100, 3, 32, 32)
        adv_dataset = TensorDataset(adv_data, adv_labels)
        loaders.append(DataLoader(adv_dataset, batch_size=10))
    return loaders


@pytest.fixture
def model() -> torch.nn.Module:
    """
    Create fake model.

    Returns
    -------
    torch.nn.Module
        Fake model.
    """
    return BasePytorchClassifier(model=MockModel())


@pytest.fixture
def data() -> torch.Tensor:
    """
    Get random samples.

    Returns
    -------
    torch.Tensor
        A fake tensor with samples.
    """
    return torch.randn(10, 3, 32, 32).clamp(0.0, 1.0)


@pytest.fixture
def labels() -> torch.Tensor:
    """
    Get random labels.

    Returns
    -------
    torch.Tensor
        A fake tensor with labels.
    """
    return torch.randint(0, 9, 10)


@pytest.fixture
def loss_values() -> torch.Tensor:
    """
    Get random model outputs.

    Returns
    -------
    torch.Tensor
        A fake tensor with model outputs.
    """
    return torch.randn(10)


@pytest.fixture
def output_values() -> torch.Tensor:
    """
    Get random model outputs.

    Returns
    -------
    torch.Tensor
        A fake tensor with model outputs.
    """
    return torch.randn(10, 10)


@pytest.fixture
def mock_hf_lm() -> HFCausalLM:
    """Create a mock Hugging Face LM wrapper without loading real weights."""
    with patch(
        "secmlt.models.hugging_face.base_hf_lm.AutoTokenizer.from_pretrained",
        return_value=MockHFTokenizer(),
    ), patch(
        "secmlt.models.hugging_face.base_hf_lm.AutoModelForCausalLM.from_pretrained",
        return_value=MockHFModel(),
    ):
        yield HFCausalLM(model_path="mock-model")
