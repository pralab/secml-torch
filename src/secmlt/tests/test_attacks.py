import pytest
from secmlt.adv.evasion.advlib_attacks.advlib_pgd import PGDAdvLib
from torch.utils.data import DataLoader
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_pgd import PGDFoolbox
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD, PGDNative


@pytest.mark.parametrize(
    "random_start",
    [True, False],
)
@pytest.mark.parametrize(
    "y_target",
    [None, 1],
)
@pytest.mark.parametrize(
    (
        "backend",
        "perturbation_models",
    ),
    [
        (
            "foolbox",
            PGDFoolbox.get_perturbation_models(),
        ),
        (
            "advlib",
            PGDAdvLib.get_perturbation_models(),
        ),
        (
            "native",
            PGDNative.get_perturbation_models(),
        ),
    ],
)
def test_pgd_attack(
    backend,
    perturbation_models,
    random_start,
    y_target,
    model,
    data_loader,
) -> BaseEvasionAttack:
    for perturbation_model in LpPerturbationModels.pert_models:
        if perturbation_model in perturbation_models:
            attack = PGD(
                perturbation_model=perturbation_model,
                epsilon=0.5,
                num_steps=10,
                step_size=0.1,
                random_start=random_start,
                y_target=y_target,
                backend=backend,
            )
            assert isinstance(attack(model, data_loader), DataLoader)
        else:
            with pytest.raises(NotImplementedError):
                attack = PGD(
                    perturbation_model=perturbation_model,
                    epsilon=0.5,
                    num_steps=10,
                    step_size=0.1,
                    random_start=random_start,
                    y_target=y_target,
                    backend=backend,
                )
