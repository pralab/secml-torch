import torch

from secmlt.metrics.classification import (
    Accuracy,
    AccuracyEnsemble,
    AttackSuccessRate,
    EnsembleSuccessRate,
)


def test_accuracy(model, data_loader) -> None:
    acc_metric = Accuracy()
    acc = acc_metric(model, data_loader)
    assert torch.is_tensor(acc)


def test_attack_success_rate(model, adv_loaders):
    attack_acc = AttackSuccessRate()
    acc = attack_acc(model, adv_loaders[0])
    assert torch.is_tensor(acc)


def test_accuracy_ensemble(model, adv_loaders):
    acc_ensemble = AccuracyEnsemble()
    acc = acc_ensemble(model, adv_loaders)
    assert torch.is_tensor(acc)


def test_ensemble_success_rate(model, adv_loaders):
    ensemble_acc = EnsembleSuccessRate()
    acc = ensemble_acc(model, adv_loaders)
    assert torch.is_tensor(acc)
