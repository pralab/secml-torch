"""Implementations of the Decoupled Direction and Norm evasion attack."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.modular_attacks.modular_attack import CE_LOSS
from secmlt.adv.evasion.modular_attacks.modular_attack_min_distance import (
    ModularEvasionAttackMinDistance,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import ClipConstraint, L2Constraint
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler


class DDN(BaseEvasionAttackCreator):
    """Creator for the Decoupled Direction and Norm (DDN) attack."""

    def __new__(
        cls,
        perturbation_model: str = LpPerturbationModels.L2,
        num_steps: int = 100,
        init_epsilon: float = 1.0,
        gamma: float = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """Create the DDN attack."""
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        implementation.check_perturbation_model_available(perturbation_model)
        return implementation(
            perturbation_model=perturbation_model,
            num_steps=num_steps,
            init_epsilon=init_epsilon,
            gamma=gamma,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the DDN attack."""
        return [Backends.FOOLBOX, Backends.ADVLIB, Backends.NATIVE]

    @staticmethod
    def _get_foolbox_implementation() -> type[DDNFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_ddn import DDNFoolbox

            return DDNFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_advlib_implementation() -> type[DDNAdvLib]:  # noqa: F821
        if importlib.util.find_spec("adv_lib", None) is not None:
            from secmlt.adv.evasion.advlib_attacks import DDNAdvLib

            return DDNAdvLib
        msg = "adv_lib extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_native_implementation() -> type[DDNNative]:
        return DDNNative


class DDNNative(ModularEvasionAttackMinDistance):
    """Native implementation of the Decoupled Direction and Norm attack."""

    def __init__(
        self,
        perturbation_model: str,
        num_steps: int,
        init_epsilon: float,
        gamma: float,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: list[Tracker] | Tracker | None = None,
        step_size: float = 1.0,
        min_step_size: float | None = 0.01,
        optimizer_cls: str | Optimizer | None = None,
        scheduler_cls: str | _LRScheduler | None = None,
    ) -> None:
        """Create the native DDN attack."""
        self.lb_value = lb
        self.ub_value = ub
        if optimizer_cls is None:
            optimizer_cls = OptimizerFactory.create_sgd(lr=step_size)
        if scheduler_cls is None:
            scheduler_cls = LRSchedulerFactory.create_cosine_annealing()

        perturbation_constraint = L2Constraint(radius=init_epsilon)
        manipulation_function = AdditiveManipulation(
            domain_constraints=[ClipConstraint(lb=lb, ub=ub)],
            perturbation_constraints=[perturbation_constraint],
        )

        gradient_processing = LinearProjectionGradientProcessing(
            LpPerturbationModels.L2
        )

        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=CE_LOSS,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            manipulation_function=manipulation_function,
            initializer=Initializer(),
            gradient_processing=gradient_processing,
            trackers=trackers,
            gamma=gamma,
            min_step_size=min_step_size,
            initial_epsilon=init_epsilon,
        )

        self.perturbation_model = LpPerturbationModels.get_p(perturbation_model)

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Return available perturbation models for the attack."""
        return {LpPerturbationModels.L2}
