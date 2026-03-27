import pytest
import torch
from secmlt.adv.evasion.advlib_attacks.advlib_pgd import PGDAdvLib
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.ddn import DDN
from secmlt.adv.evasion.fmn import FMN, FMNNative
from secmlt.adv.evasion.foolbox_attacks.foolbox_pgd import PGDFoolbox
from secmlt.adv.evasion.modular_attacks.modular_attack import (
    CE_LOSS,
    ModularEvasionAttack,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD, PGDNative
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.trackers.trackers import LossTracker
from torch.utils.data import DataLoader

from src.secmlt.adv.evasion.advlib_attacks.advlib_fmn import FMNAdvLib
from src.secmlt.adv.evasion.foolbox_attacks.foolbox_fmn import FMNFoolbox
from src.secmlt.adv.evasion.modular_attacks.eot_gradient import EoTGradientMixin

PGDEoT = type("PGDEoT", (EoTGradientMixin, PGDNative), {})


class IdentityGradientProcessingMock(GradientProcessing):
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        return grad


class DummyModularAttack(ModularEvasionAttack):
    def _run_loop(
        self,
        model,
        delta: torch.Tensor,
        samples: torch.Tensor,
        target: torch.Tensor,
        optimizer,
        scheduler,
        multiplier: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return samples, delta


def no_scheduler(*args, **kwargs):
    return None


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


def test_attack_can_return_generator(model, data_loader):
    attack = PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.5,
        num_steps=3,
        step_size=0.1,
        random_start=False,
        y_target=None,
        backend="native",
    )

    batch_iterator = attack(model, data_loader, stream=True)
    assert not isinstance(batch_iterator, DataLoader)

    attacked_batches = list(batch_iterator)
    assert len(attacked_batches) == len(data_loader)

    total_attacked_samples = sum(batch_adv.shape[0]
                                for batch_adv, _ in attacked_batches)
    total_labels = sum(batch_labels.shape[0] for _, batch_labels in attacked_batches)
    assert total_attacked_samples == len(data_loader.dataset)
    assert total_labels == len(data_loader.dataset)


def test_attack_warns_when_streaming_with_trackers(model, data_loader):
    attack = PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.5,
        num_steps=3,
        step_size=0.1,
        random_start=False,
        y_target=None,
        backend="native",
        trackers=[LossTracker()],
    )

    with pytest.warns(UserWarning, match="Trackers are enabled while streaming"):
        batch_iterator = attack(model, data_loader, stream=True)

    first_batch_adv, first_batch_labels = next(batch_iterator)
    assert first_batch_adv.shape[0] == first_batch_labels.shape[0]


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


def test_modular_attack_raises_for_unknown_loss_name():
    with pytest.raises(ValueError, match="Loss function not found"):
        DummyModularAttack(
            y_target=None,
            num_steps=1,
            step_size=0.1,
            loss_function="unknown_loss",
            optimizer_cls=torch.optim.SGD,
            scheduler_cls=no_scheduler,
            manipulation_function=AdditiveManipulation([], []),
            initializer=Initializer(),
            gradient_processing=IdentityGradientProcessingMock(),
        )


def test_modular_attack_sets_loss_from_string_name():
    attack = DummyModularAttack(
        y_target=None,
        num_steps=1,
        step_size=0.1,
        loss_function=CE_LOSS,
        optimizer_cls=torch.optim.SGD,
        scheduler_cls=no_scheduler,
        manipulation_function=AdditiveManipulation([], []),
        initializer=Initializer(),
        gradient_processing=IdentityGradientProcessingMock(),
    )

    assert isinstance(attack.loss_function, torch.nn.CrossEntropyLoss)
    assert attack.loss_function.reduction == "none"


def test_modular_attack_accepts_custom_loss_instance():
    attack = DummyModularAttack(
        y_target=None,
        num_steps=1,
        step_size=0.1,
        loss_function=torch.nn.CrossEntropyLoss(),
        optimizer_cls=torch.optim.SGD,
        scheduler_cls=no_scheduler,
        manipulation_function=AdditiveManipulation([], []),
        initializer=Initializer(),
        gradient_processing=IdentityGradientProcessingMock(),
    )

    custom_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    attack.loss_function = custom_loss

    assert attack.loss_function is custom_loss
    assert attack.loss_function.reduction == "mean"
