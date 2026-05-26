import pytest
from secmlt.adv.jailbreak.base_jailbreak_attack import BaseJailbreakAttack


def dummy_is_jailbreak(_model, _behavior, _adv_prompt, logs):
    return logs["success"]


class DummyAttack(BaseJailbreakAttack):
    def __init__(self):
        super().__init__(is_jailbreak=dummy_is_jailbreak)
        self.run_calls = 0

    def _run(self, model, behavior, objective=None):
        self.run_calls += 1
        return "adv_prompt", {"success": behavior["success"], "objective": objective}


class DummyAttackWithoutCriterion(BaseJailbreakAttack):
    def _run(self, model, behavior, objective=None):
        return "adv_prompt", {}


def test_base_jailbreak_call_returns_list(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [
        {
            "BehaviorID": "0",
            "Behavior": "test",
            "ContextString": "",
            "success": True,
        },
        {
            "BehaviorID": "1",
            "Behavior": "test2",
            "ContextString": "",
            "success": False,
        },
    ]

    out = attack(mock_hf_lm, behaviors)

    assert isinstance(out, list)
    assert len(out) == len(behaviors)


def test_base_jailbreak_call_returns_tuple_elements(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [
        {
            "BehaviorID": "0",
            "Behavior": "test",
            "ContextString": "",
            "success": True,
        },
    ]
    out = attack(mock_hf_lm, behaviors)

    adv_prompt, logs = out[0]
    assert isinstance(adv_prompt, str)
    assert isinstance(logs, dict)


def test_base_jailbreak_call_runs_each_behavior(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [
        {
            "BehaviorID": "0",
            "Behavior": "test",
            "ContextString": "",
            "success": True,
        },
        {
            "BehaviorID": "1",
            "Behavior": "test2",
            "ContextString": "",
            "success": False,
        },
    ]
    attack(mock_hf_lm, behaviors)

    assert attack.run_calls == len(behaviors)


def test_base_jailbreak_forwards_objectives(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [
        {
            "BehaviorID": "0",
            "Behavior": "test",
            "ContextString": "",
            "success": True,
        },
        {
            "BehaviorID": "1",
            "Behavior": "test2",
            "ContextString": "",
            "success": False,
        },
    ]
    objectives = ["Sure, here is", "Absolutely, here are"]

    out = attack(mock_hf_lm, behaviors, objectives=objectives)

    assert [logs["objective"] for _, logs in out] == objectives


def test_base_jailbreak_raises_on_mismatched_objectives(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [
        {
            "BehaviorID": "0",
            "Behavior": "test",
            "ContextString": "",
            "success": True,
        },
    ]

    with pytest.raises(ValueError, match="same length"):
        attack(mock_hf_lm, behaviors, objectives=["objective", "extra"])


def test_base_jailbreak_uses_configured_success_criterion(mock_hf_lm):
    attack = DummyAttack()

    behavior = {
        "BehaviorID": "0",
        "Behavior": "test",
        "ContextString": "",
        "success": True,
    }
    out = attack(mock_hf_lm, [behavior])
    adv_prompt, logs = out[0]

    assert attack.is_jailbreak(mock_hf_lm, behavior, adv_prompt, logs)


def test_base_jailbreak_raises_without_success_criterion(mock_hf_lm):
    attack = DummyAttackWithoutCriterion()

    with pytest.raises(NotImplementedError):
        attack.is_jailbreak(mock_hf_lm, {}, "adv_prompt", {})


def test_base_jailbreak_base_raises_if_not_implemented(mock_hf_lm):
    attack = BaseJailbreakAttack()

    behaviors = [{"BehaviorID": "0", "Behavior": "test", "ContextString": ""}]

    with pytest.raises(NotImplementedError):
        attack(mock_hf_lm, behaviors)
