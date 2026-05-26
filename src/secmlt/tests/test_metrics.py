import pytest
import torch
from secmlt.metrics.classification import (
    Accuracy,
    AccuracyEnsemble,
    AttackSuccessRate,
    EnsembleSuccessRate,
    JailbreakAccuracy,
    accuracy,
)


def always_jailbreak(*_args):
    return True


def test_accuracy_function() -> None:
    y_pred = torch.tensor([True, False])
    y_true = torch.tensor([True, True])

    acc = accuracy(y_pred, y_true)

    assert torch.is_tensor(acc)
    assert acc == torch.tensor(0.5)


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


def test_jailbreak_accuracy(mock_hf_lm):
    class DummyAttack:
        def is_jailbreak(self, model, behavior, adv_prompt, logs):
            return logs["success"]

    behaviors = [
        {"BehaviorID": "0", "Behavior": "test", "ContextString": ""},
        {"BehaviorID": "1", "Behavior": "test2", "ContextString": ""},
    ]
    attack_results = [
        ("adv_prompt", {"success": True}),
        ("adv_prompt", {"success": False}),
    ]
    attack = DummyAttack()
    metric = JailbreakAccuracy(is_jailbreak=attack.is_jailbreak)

    acc = metric(mock_hf_lm, behaviors, attack_results)

    assert isinstance(metric, Accuracy)
    assert torch.is_tensor(acc)
    assert acc == torch.tensor(0.5)


def test_jailbreak_accuracy_uses_logged_success_without_judge(mock_hf_lm):
    class DummyAttack:
        def is_jailbreak(self, model, behavior, adv_prompt, logs):
            return logs["is_jailbreak"]

    behaviors = [
        {"BehaviorID": "0", "Behavior": "test", "ContextString": ""},
        {"BehaviorID": "1", "Behavior": "test2", "ContextString": ""},
    ]
    attack_results = [
        ("adv_prompt", {"is_jailbreak": True}),
        ("adv_prompt", {"is_jailbreak": False}),
    ]
    attack = DummyAttack()
    metric = JailbreakAccuracy(is_jailbreak=attack.is_jailbreak)

    acc = metric(mock_hf_lm, behaviors, attack_results)

    assert torch.is_tensor(acc)
    assert acc == torch.tensor(0.5)


def test_jailbreak_accuracy_raises_on_mismatched_batches(mock_hf_lm):
    metric = JailbreakAccuracy(is_jailbreak=always_jailbreak)

    with pytest.raises(ValueError, match="same length"):
        metric(mock_hf_lm, [], [("adv_prompt", {})])


def test_jailbreak_accuracy_raises_on_empty_batch(mock_hf_lm):
    metric = JailbreakAccuracy(is_jailbreak=always_jailbreak)

    with pytest.raises(ValueError, match="empty batch"):
        metric(mock_hf_lm, [], [])
