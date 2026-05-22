import pytest
from secmlt.adv.jailbreak.base_jailbreak_attack import BaseJailbreakAttack


class DummyAttack(BaseJailbreakAttack):
    def _run(self, model, behavior, objective=None):
        return "adv_prompt", {}

    def is_jailbreak(self, model, behavior, adv_prompt, logs, objective=None):
        return True


def test_base_jailbreak_call_returns_list(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [
        {"BehaviorID": "0", "Behavior": "test", "ContextString": ""},
        {"BehaviorID": "1", "Behavior": "test2", "ContextString": ""},
    ]

    out = attack(mock_hf_lm, behaviors)

    assert isinstance(out, list)
    assert len(out) == len(behaviors)


def test_base_jailbreak_call_returns_tuple_elements(mock_hf_lm):
    attack = DummyAttack()

    behaviors = [{"BehaviorID": "0", "Behavior": "test", "ContextString": ""}]
    out = attack(mock_hf_lm, behaviors)

    adv_prompt, logs = out[0]
    assert isinstance(adv_prompt, str)
    assert isinstance(logs, dict)


def test_base_jailbreak_base_raises_if_not_implemented(mock_hf_lm):
    attack = BaseJailbreakAttack()

    behaviors = [{"BehaviorID": "0", "Behavior": "test", "ContextString": ""}]

    with pytest.raises(NotImplementedError):
        attack(mock_hf_lm, behaviors)
