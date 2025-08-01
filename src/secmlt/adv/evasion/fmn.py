"""Implementations of the Fast Minimum-Norm evasion attack."""

from __future__ import annotations  # noqa: I001

import importlib.util

import torch

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.modular_attacks.modular_attack import (
    LOGIT_LOSS,
)
from secmlt.adv.evasion.modular_attacks.modular_attack_min_distance import (
    ModularEvasionAttackMinDistance,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    ClipConstraint,
    L0Constraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class FMN(BaseEvasionAttackCreator):
    """Creator for the Fast Minimum-Norm (FMN) attack."""

    def __new__(
        cls,
        perturbation_model: str,
        num_steps: int,
        step_size: float,
        min_step_size: float | None = None,
        gamma: float = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.NATIVE,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the FMN attack.

        Parameters
        ----------
        perturbation_model : str
            The perturbation model to be used for the attack.
        num_steps : int
            The number of iterations for the attack.
        max_step_size : float
            The attack step size.
        min_step_size : float, optional
            The number of attack restarts. The default value is 1.
        gamma: float, optional
            Step size for modifying the eps-ball. Will decay with cosine annealing.
        y_target : int | None, optional
            The target label for the attack. If None, the attack is
            untargeted. The default value is None.
        lb : float, optional
            The lower bound for the perturbation. The default value is 0.0.
        ub : float, optional
            The upper bound for the perturbation. The default value is 1.0.
        backend : str, optional
            Backend to use to run the attack, by default Backends.FOOLBOX
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            available only for native implementation, by default None.

        Returns
        -------
        BaseEvasionAttack
            FMN attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        implementation.check_perturbation_model_available(perturbation_model)
        return implementation(
            perturbation_model=perturbation_model,
            num_steps=num_steps,
            max_step_size=step_size,
            min_step_size=min_step_size,
            gamma=gamma,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the FMN attack."""
        return [Backends.FOOLBOX, Backends.ADVLIB, Backends.NATIVE]

    @staticmethod
    def _get_foolbox_implementation() -> type[FMNFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_fmn import FMNFoolbox

            return FMNFoolbox
        msg = "Foolbox extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_advlib_implementation() -> type[FMNAdvLib]:  # noqa: F821
        if importlib.util.find_spec("adv_lib", None) is not None:
            from secmlt.adv.evasion.advlib_attacks import FMNAdvLib

            return FMNAdvLib
        msg = "Adversarial Library extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_native_implementation() -> type[FMNNative]:
        return FMNNative


class FMNNative(ModularEvasionAttackMinDistance):
    """Native implementation of the Fast Minimum-Norm attack."""

    def __init__(
        self,
        perturbation_model: str,
        num_steps: int,
        max_step_size: float,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> None:
        """
        Create Native FMN attack.

        Parameters
        ----------
        perturbation_model : str
            The perturbation model to be used for the attack.
        num_steps : int
            The number of iterations for the attack.
        max_step_size : float
            The attack step size.
        min_step_size : float, optional
            The number of attack restarts. The default value is 1.
        gamma: float, optional
            Step size for modifying the eps-ball. Will decay with cosine annealing.
        y_target : int | None, optional
            The target label for the attack. If None, the attack is
            untargeted. The default value is None.
        lb : float, optional
            The lower bound for the perturbation. The default value is 0.0.
        ub : float, optional
            The upper bound for the perturbation. The default value is 1.0.
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            available only for native implementation, by default None.
        """
        perturbation_models = {
            LpPerturbationModels.L0: L0Constraint,
            LpPerturbationModels.L1: L1Constraint,
            LpPerturbationModels.L2: L2Constraint,
            LpPerturbationModels.LINF: LInfConstraint,
        }

        initializer = Initializer()
        gradient_processing = LinearProjectionGradientProcessing(
            LpPerturbationModels.L2
        )
        perturbation_constraints = [
            perturbation_models[perturbation_model](radius=torch.inf)
        ]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )

        self.perturbation_model = LpPerturbationModels.get_p(perturbation_model)
        self.perturbation_model_dual = LpPerturbationModels.get_dual(perturbation_model)

        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=max_step_size,
            loss_function=LOGIT_LOSS,
            optimizer_cls=OptimizerFactory.create_sgd(lr=max_step_size),
            scheduler_cls=LRSchedulerFactory.create_cosine_annealing(),
            manipulation_function=manipulation_function,
            gradient_processing=gradient_processing,
            initializer=initializer,
            trackers=trackers,
        )

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        """
        Check if a given perturbation model is implemented.

        Returns
        -------
        set[str]
            Set of perturbation models available for this attack.
        """
        return {
            LpPerturbationModels.L0,
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }
