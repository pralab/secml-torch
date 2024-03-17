import torch
from secmlt.adv.evasion.aggregators.ensemble import (
    FixedEpsilonEnsemble,
    MinDistanceEnsemble,
)
from secmlt.tests.mocks import MockModel


def test_min_distance_ensemble(data_loader, adv_loaders) -> None:
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


def test_fixed_epsilon_ensemble(data_loader, adv_loaders) -> None:
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
