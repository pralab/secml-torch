import importlib.metadata

import pytest
import torch
from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.advlib_attacks.advlib_fgsm import FGSMAdvLib
from secmlt.adv.evasion.advlib_attacks.advlib_fmn import FMNAdvLib
from secmlt.adv.evasion.advlib_attacks.advlib_pgd import PGDAdvLib
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.boundary_attack import BoundaryAttack
from secmlt.adv.evasion.cw import CW
from secmlt.adv.evasion.ddn import DDN
from secmlt.adv.evasion.deepfool import DeepFool
from secmlt.adv.evasion.fgsm import FGSM
from secmlt.adv.evasion.fmn import FMN, FMNNative
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_boundary import BoundaryAttackFoolbox
from secmlt.adv.evasion.foolbox_attacks.foolbox_fgsm import FGSMFoolbox
from secmlt.adv.evasion.foolbox_attacks.foolbox_fmn import FMNFoolbox
from secmlt.adv.evasion.foolbox_attacks.foolbox_hopskipjump import HopSkipJumpFoolbox
from secmlt.adv.evasion.foolbox_attacks.foolbox_pgd import PGDFoolbox
from secmlt.adv.evasion.foolbox_attacks.foolbox_spatial import SpatialAttackFoolbox
from secmlt.adv.evasion.foolbox_attacks.foolbox_vat import VATFoolbox
from secmlt.adv.evasion.hopskipjump import HopSkipJump
from secmlt.adv.evasion.modular_attacks.eot_gradient import EoTGradientMixin
from secmlt.adv.evasion.modular_attacks.modular_attack import (
    CE_LOSS,
    ModularEvasionAttack,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD, PGDNative
from secmlt.adv.evasion.spatial_attack import SpatialAttack
from secmlt.adv.evasion.vat import VAT
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.trackers.trackers import LossTracker
from torch.utils.data import DataLoader

PGDEoT = type("PGDEoT", (EoTGradientMixin, PGDNative), {})


def _adv_lib_gte(major: int, minor: int, patch: int) -> bool:
    try:
        version_str = importlib.metadata.version("adv-lib")
    except importlib.metadata.PackageNotFoundError:
        return False
    else:
        parts = tuple(int(x) for x in version_str.split(".")[:3])
        return parts >= (major, minor, patch)


adv_lib_gte_023 = _adv_lib_gte(0, 2, 3)


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


def test_advlib_base_trackers_allowed():
    assert BaseAdvLibEvasionAttack._trackers_allowed() is True


def test_foolbox_base_trackers_allowed():
    assert BaseFoolboxEvasionAttack._trackers_allowed() is True


def test_advlib_base_raises_on_unsupported_model(data_loader):
    attack = BaseAdvLibEvasionAttack(advlib_attack=lambda **kwargs: kwargs["inputs"])
    samples, labels = next(iter(data_loader))

    with pytest.raises(NotImplementedError, match="Model type not supported"):
        attack._run(model=object(), samples=samples, labels=labels)


def test_foolbox_base_raises_on_unsupported_model(data_loader):
    attack = BaseFoolboxEvasionAttack(
        foolbox_attack=lambda **kwargs: (None, kwargs["inputs"], None),
    )
    samples, labels = next(iter(data_loader))

    with pytest.raises(NotImplementedError, match="Model type not supported"):
        attack._run(model=object(), samples=samples, labels=labels)


def test_advlib_base_targeted_and_epsilon_kwargs(model, data_loader):
    calls = {}

    def _fake_advlib_attack(**kwargs):
        calls.update(kwargs)
        return kwargs["inputs"] + 0.1

    attack = BaseAdvLibEvasionAttack(
        advlib_attack=_fake_advlib_attack,
        epsilon=0.2,
        y_target=3,
    )
    samples, labels = next(iter(data_loader))

    advx, delta = attack._run(model=model, samples=samples, labels=labels)

    assert calls["targeted"] is True
    assert torch.equal(calls["labels"], torch.full_like(labels, 3))
    assert calls["ε"] == pytest.approx(0.2)
    assert torch.allclose(delta, advx - samples.to(advx.device))


def test_foolbox_base_untargeted_and_targeted_criteria(monkeypatch, model, data_loader):
    class _Criterion:
        def __init__(self, labels):
            self.labels = labels

    class _TargetedCriterion:
        def __init__(self, labels):
            self.labels = labels

    class _DummyFBModel:
        def __init__(self, wrapped_model, bounds, device):
            self.wrapped_model = wrapped_model
            self.bounds = bounds
            self.device = device

    monkeypatch.setattr(
        "secmlt.adv.evasion.foolbox_attacks.foolbox_base.Misclassification",
        _Criterion,
    )
    monkeypatch.setattr(
        "secmlt.adv.evasion.foolbox_attacks.foolbox_base.TargetedMisclassification",
        _TargetedCriterion,
    )
    monkeypatch.setattr(
        "secmlt.adv.evasion.foolbox_attacks.foolbox_base.PyTorchModel",
        _DummyFBModel,
    )

    calls = []

    def _fake_foolbox_attack(**kwargs):
        calls.append(kwargs)
        return None, kwargs["inputs"] + 0.25, None

    samples, labels = next(iter(data_loader))

    untargeted = BaseFoolboxEvasionAttack(
        foolbox_attack=_fake_foolbox_attack,
        epsilon=0.5,
        y_target=None,
    )
    advx_u, delta_u = untargeted._run(model=model, samples=samples, labels=labels)
    assert isinstance(calls[-1]["criterion"], _Criterion)
    assert torch.equal(calls[-1]["criterion"].labels, labels.to(model._get_device()))
    assert calls[-1]["epsilons"] == pytest.approx(0.5)
    assert torch.allclose(delta_u, advx_u - samples.to(advx_u.device))

    targeted = BaseFoolboxEvasionAttack(
        foolbox_attack=_fake_foolbox_attack,
        epsilon=0.3,
        y_target=2,
    )
    advx_t, delta_t = targeted._run(model=model, samples=samples, labels=labels)
    assert isinstance(calls[-1]["criterion"], _TargetedCriterion)
    assert torch.equal(calls[-1]["criterion"].labels, torch.full_like(labels, 2))
    assert torch.allclose(delta_t, advx_t - samples.to(advx_t.device))


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
def test_fgsm_foolbox_attack(
    y_target,
    model,
    data_loader,
) -> None:
    for perturbation_model in LpPerturbationModels.pert_models:
        if perturbation_model in FGSMFoolbox.get_perturbation_models():
            attack = FGSMFoolbox(
                perturbation_model=perturbation_model,
                epsilon=0.1,
                y_target=y_target,
            )
            assert isinstance(attack(model, data_loader), DataLoader)


@pytest.mark.parametrize(
    "y_target",
    [None, 1],
)
def test_fgsm_advlib_attack(
    y_target,
    model,
    data_loader,
) -> None:
    for perturbation_model in LpPerturbationModels.pert_models:
        if perturbation_model in FGSMAdvLib.get_perturbation_models():
            attack = FGSMAdvLib(
                perturbation_model=perturbation_model,
                epsilon=0.1,
                y_target=y_target,
            )
            assert isinstance(attack(model, data_loader), DataLoader)


def test_fgsm_advlib_raises_on_invalid_loss():
    with pytest.raises(ValueError, match="FGSM AdvLib supports only these losses"):
        FGSMAdvLib(
            perturbation_model=LpPerturbationModels.LINF,
            epsilon=0.1,
            loss_function="unknown",
        )


@pytest.mark.parametrize(
    "y_target",
    [None, 1],
)
@pytest.mark.parametrize(
    ("backend", "perturbation_models_fgsm"),
    [
        ("foolbox", FGSMFoolbox.get_perturbation_models()),
        ("advlib", FGSMAdvLib.get_perturbation_models()),
    ],
)
def test_fgsm_attack(
    backend,
    perturbation_models_fgsm,
    y_target,
    model,
    data_loader,
) -> None:
    for perturbation_model in LpPerturbationModels.pert_models:
        if perturbation_model in perturbation_models_fgsm:
            attack = FGSM(
                perturbation_model=perturbation_model,
                epsilon=0.1,
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

    total_attacked_samples = sum(
        batch_adv.shape[0] for batch_adv, _ in attacked_batches
    )
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


def test_attack_accepts_raw_nn_module(data_loader):
    """Raw nn.Module (no decision_function) must be auto-wrapped by __call__."""

    class RawModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    attack = PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.3,
        num_steps=3,
        step_size=0.1,
        random_start=False,
        backend="native",
    )
    assert isinstance(attack(RawModel(), data_loader), DataLoader)


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


@pytest.mark.parametrize(
    "y_target",
    [None, 1],
)
@pytest.mark.parametrize(
    ("backend",),
    [
        ("foolbox",),
        ("advlib",),
    ],
)
def test_cw_attack(
    backend,
    y_target,
    model,
    data_loader,
) -> None:
    attack = CW(
        binary_search_steps=2,
        num_steps=10,
        step_size=0.01,
        confidence=0.0,
        initial_const=0.001,
        y_target=y_target,
        backend=backend,
    )
    assert isinstance(attack(model, data_loader), DataLoader)


@pytest.mark.parametrize(
    ("backend",),
    [
        ("foolbox",),
        pytest.param(
            "advlib",
            marks=pytest.mark.skipif(
                not adv_lib_gte_023,
                reason="adv_lib >= 0.2.3 required for DeepFool advlib backend",
            ),
        ),
    ],
)
def test_deepfool_attack(
    backend,
    model,
    data_loader,
) -> None:
    attack = DeepFool(
        num_steps=10,
        overshoot=0.02,
        backend=backend,
    )
    assert isinstance(attack(model, data_loader), DataLoader)


def test_vat_foolbox_attack(model, data_loader) -> None:
    attack = VATFoolbox(epsilon=0.1, steps=1)
    assert isinstance(attack(model, data_loader), DataLoader)


def test_vat_attack(model, data_loader) -> None:
    attack = VAT(epsilon=0.1, steps=1, backend="foolbox")
    assert isinstance(attack(model, data_loader), DataLoader)


@pytest.fixture
def deterministic_model() -> BasePyTorchClassifier:
    """Simple deterministic model for decision-based attack tests."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
    )
    net.eval()
    return BasePyTorchClassifier(model=net)


@pytest.mark.parametrize(
    "y_target",
    # y_target=2 chosen because class 2 dominates uniform-noise predictions for
    # the deterministic_model fixture, making init_attack reliable.
    [None, 2],
)
def test_boundary_attack_foolbox(y_target, deterministic_model, data_loader) -> None:
    attack = BoundaryAttackFoolbox(steps=100, y_target=y_target)
    assert isinstance(attack(deterministic_model, data_loader), DataLoader)


@pytest.mark.parametrize(
    "y_target",
    [None, 2],
)
def test_boundary_attack(y_target, deterministic_model, data_loader) -> None:
    attack = BoundaryAttack(steps=100, y_target=y_target, backend="foolbox")
    assert isinstance(attack(deterministic_model, data_loader), DataLoader)


def test_boundary_attack_only_foolbox_backend() -> None:
    with pytest.raises(
        NotImplementedError, match="Unsupported or not-implemented backend"
    ):
        BoundaryAttack(steps=100, backend="native")


@pytest.mark.parametrize("y_target", [None, 2])
@pytest.mark.parametrize(
    "perturbation_model",
    [LpPerturbationModels.L2, LpPerturbationModels.LINF],
)
def test_hopskipjump_foolbox(
    y_target, perturbation_model, deterministic_model, data_loader
) -> None:
    attack = HopSkipJumpFoolbox(
        perturbation_model=perturbation_model,
        steps=5,
        initial_gradient_eval_steps=10,
        max_gradient_eval_steps=20,
        y_target=y_target,
    )
    assert isinstance(attack(deterministic_model, data_loader), DataLoader)


@pytest.mark.parametrize("y_target", [None, 2])
@pytest.mark.parametrize(
    "perturbation_model",
    [LpPerturbationModels.L2, LpPerturbationModels.LINF],
)
def test_hopskipjump_attack(
    y_target, perturbation_model, deterministic_model, data_loader
) -> None:
    attack = HopSkipJump(
        perturbation_model=perturbation_model,
        steps=5,
        initial_gradient_eval_steps=10,
        max_gradient_eval_steps=20,
        y_target=y_target,
        backend="foolbox",
    )
    assert isinstance(attack(deterministic_model, data_loader), DataLoader)


def test_hopskipjump_only_foolbox_backend() -> None:
    with pytest.raises(
        NotImplementedError, match="Unsupported or not-implemented backend"
    ):
        HopSkipJump(steps=5, backend="native")


@pytest.mark.parametrize("y_target", [None, 2])
@pytest.mark.parametrize("grid_search", [True, False])
def test_spatial_attack_foolbox(
    y_target, grid_search, deterministic_model, data_loader
) -> None:
    attack = SpatialAttackFoolbox(
        max_translation=2,
        max_rotation=15,
        num_translations=3,
        num_rotations=3,
        grid_search=grid_search,
        random_steps=5,
        y_target=y_target,
    )
    assert isinstance(attack(deterministic_model, data_loader), DataLoader)


@pytest.mark.parametrize("y_target", [None, 2])
def test_spatial_attack(y_target, deterministic_model, data_loader) -> None:
    attack = SpatialAttack(
        max_translation=2,
        max_rotation=15,
        num_translations=3,
        num_rotations=3,
        y_target=y_target,
        backend="foolbox",
    )
    assert isinstance(attack(deterministic_model, data_loader), DataLoader)


def test_spatial_attack_only_foolbox_backend() -> None:
    with pytest.raises(
        NotImplementedError, match="Unsupported or not-implemented backend"
    ):
        SpatialAttack(backend="native")


def test_spatial_attack_no_lp_perturbation_models() -> None:
    assert SpatialAttackFoolbox.get_perturbation_models() == set()
