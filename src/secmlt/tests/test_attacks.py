import pytest
from secmlt.adv.evasion.advlib_attacks.advlib_pgd import PGDAdvLib
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.ddn import DDN
from secmlt.adv.evasion.fmn import FMN, FMNNative
from secmlt.adv.evasion.foolbox_attacks.foolbox_pgd import PGDFoolbox
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD, PGDNative
from torch.utils.data import DataLoader

from src.secmlt.adv.evasion.advlib_attacks.advlib_fmn import FMNAdvLib
from src.secmlt.adv.evasion.foolbox_attacks.foolbox_fmn import FMNFoolbox
from src.secmlt.adv.evasion.modular_attacks.eot_gradient import EoTGradientMixin

PGDEoT = type("PGDEoT", (EoTGradientMixin, PGDNative), {})


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
        "perturbation_models_pgd",
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
    perturbation_models_pgd,
    random_start,
    y_target,
    model,
    data_loader,
) -> BaseEvasionAttack:
    for perturbation_model in LpPerturbationModels.pert_models:
        if perturbation_model in perturbation_models_pgd:
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


def test_eot_mixin(model, data_loader):
    """Test that the EoT mixin can be added to PGD and works as expected."""
    k = 3
    radius = 0.02

    # take a tiny subset for speed

    attack = PGDEoT(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.3,
        num_steps=3,  # keep it cheap
        step_size=0.1,
        random_start=True,
        y_target=None,
        backend="native",
        eot_samples=k,
        eot_radius=radius,
    )

    # check inheritance
    assert isinstance(attack, PGDNative)
    assert isinstance(attack, EoTGradientMixin)

    # check parameters
    assert attack.eot_samples == k
    assert attack.eot_radius == radius

    # run attack
    adv_loader = attack(model, data_loader)
    assert isinstance(adv_loader, DataLoader)


@pytest.mark.parametrize(
    "y_target",
    [None, 1],
)
@pytest.mark.parametrize(
    (
        "backend",
        "perturbation_models_fmn",
    ),
    [
        (
            "foolbox",
            FMNFoolbox.get_perturbation_models(),
        ),
        (
            "advlib",
            FMNAdvLib.get_perturbation_models(),
        ),
        (
            "native",
            FMNNative.get_perturbation_models(),
        ),
    ],
)
def test_fmn_attack(
    backend,
    perturbation_models_fmn,
    y_target,
    model,
    data_loader,
) -> BaseEvasionAttack:
    for perturbation_model in LpPerturbationModels.pert_models:
        if perturbation_model in perturbation_models_fmn:
            # skip test for L0 adv lib implementation untargeted attack
            # due to a problem when attack is succesful with zero
            # perturbation in all samples
            if (
                backend == "advlib"
                and perturbation_model == LpPerturbationModels.L0
                and y_target is None
            ):
                continue
            attack = FMN(
                perturbation_model=perturbation_model,
                num_steps=10,
                step_size=0.1,
                y_target=y_target,
                backend=backend,
            )
            assert isinstance(attack(model, data_loader), DataLoader)


@pytest.mark.parametrize(
    "y_target",
    [None, 1],
)
@pytest.mark.parametrize(
    ("backend",),
    [
        ("foolbox",),
        ("advlib",),
        ("native",),
    ],
)
def test_ddn_attack(
    backend,
    y_target,
    model,
    data_loader,
) -> BaseEvasionAttack:
    attack = DDN(
        num_steps=5,
        eps_init=1.0,
        gamma=0.05,
        y_target=y_target,
        backend=backend,
    )
    assert isinstance(attack(model, data_loader), DataLoader)


@pytest.mark.parametrize("attack_class", [PGD, FMN])
def test_invalid_perturbation_models(attack_class):
    """Test that an error is raised for invalid perturbation models."""

    common_args = {
        "perturbation_model": "invalid_perturbation_model",
        "num_steps": 10,
        "step_size": 0.1,
        "backend": "native",  # Backends.NATIVE
    }

    if attack_class is PGD:
        common_args["epsilon"] = 0.5

    with pytest.raises(
        NotImplementedError, match=r"Unsupported or not-implemented perturbation model."
    ):
        attack_class(**common_args)
