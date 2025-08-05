import pytest
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


@pytest.mark.parametrize(
    "model, expected_p, expected_dual",
    [
        ("l0", 0, None),
        ("l1", 1, float("inf")),
        ("l2", 2, 2),
        ("linf", float("inf"), 1),
    ],
)
def test_valid_perturbation_models(model, expected_p, expected_dual):
    assert LpPerturbationModels.is_perturbation_model_available(model) is True
    assert LpPerturbationModels.get_p(model) == expected_p
    assert LpPerturbationModels.get_dual(model) == expected_dual


@pytest.mark.parametrize("model", ["l3", "", None, 123])
def test_invalid_perturbation_models(model):
    assert not LpPerturbationModels.is_perturbation_model_available(model)

    with pytest.raises(ValueError, match="Perturbation model not implemented"):
        LpPerturbationModels.get_p(model)

    with pytest.raises(ValueError, match="Perturbation model not implemented"):
        LpPerturbationModels.get_dual(model)
