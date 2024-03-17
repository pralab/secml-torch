import pytest
import torch
from secmlt.adv.evasion.aggregators.ensemble import (
    FixedEpsilonEnsemble,
    MinDistanceEnsemble,
)
from torch.utils.data import DataLoader, TensorDataset


class MockModel(torch.nn.Module):
    def forward(self, x):
        # Mock output shape (batch_size, 10)
        return torch.randn(x.size(0), 10)


@pytest.fixture()
def data_loader():
    # Create a dummy dataset loader for testing
    data = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture()
def adv_loaders():
    # Create a list of dummy adversarial example loaders for testing
    adv_data = torch.randn(100, 3, 32, 32)
    adv_labels = torch.randint(0, 10, (100,))
    adv_dataset = TensorDataset(adv_data, adv_labels)
    return [DataLoader(adv_dataset, batch_size=10) for _ in range(3)]


def test_min_distance_ensemble(data_loader, adv_loaders):
    model = MockModel()
    ensemble = MinDistanceEnsemble("l2")
    result_loader = ensemble(model, data_loader, adv_loaders)
    for batch in result_loader:
        assert batch[0].shape == (
            10,
            3,
            32,
            32,
        )  # Expected shape of adversarial examples
        assert batch[1].shape == (10,)  # Expected shape of original labels


def test_fixed_epsilon_ensemble(data_loader, adv_loaders):
    model = MockModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    ensemble = FixedEpsilonEnsemble(loss_fn)
    result_loader = ensemble(model, data_loader, adv_loaders)
    for batch in result_loader:
        assert batch[0].shape == (
            10,
            3,
            32,
            32,
        )  # Expected shape of adversarial examples
        assert batch[1].shape == (10,)  # Expected shape of original labels
