"""Decoupled Direction and Norm (DDN) attack implementation."""

from __future__ import annotations  # noqa: I001

import importlib.util

import torch

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.modular_attacks.modular_attack import (
    CE_LOSS,
)
from secmlt.adv.evasion.modular_attacks.modular_attack_min_distance import (
    ModularEvasionAttackMinDistance,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    ClipConstraint,
    L2Constraint,
)
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class DDN(BaseEvasionAttackCreator):
    """Creator for the Decoupled Direction and Norm (DDN) attack."""

    def __new__(
        cls,
        num_steps: int,
        eps_init: float = 8 / 255,
        gamma: float = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.NATIVE,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the DDN attack.

        References
        ----------
        .. [#Rony18] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira,
            Ismail Ben Ayed, Robert Sabourin, Eric Granger, "Decoupling
            Direction and Norm for Efficient Gradient-Based L2 Adversarial
            Attacks and Defenses", https://arxiv.org/abs/1811.09600

        Parameters
        ----------
        num_steps : int
            The number of iterations for the attack.
        eps_init: float, optional
            Initial L2 norm of the perturbation. The default value is 8/255.
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
            DDN attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            num_steps=num_steps,
            eps_init=eps_init,
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
    """Native implementation of the Decoupled Direction and Norm (DDN) attack."""

    def __init__(
        self,
        num_steps: int,
        eps_init: float = 8 / 255,
        gamma: float = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> None:
        """
        Create Native DDN attack.

        Parameters
        ----------
        num_steps : int
            The number of iterations for the attack.
        eps_init: float, optional
            Initial L2 norm of the perturbation. The default value is 8/255.
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
        initializer = Initializer()
        gradient_processing = LinearProjectionGradientProcessing(
            LpPerturbationModels.L2
        )
        perturbation_constraints = [L2Constraint(radius=eps_init)]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )

        self.perturbation_model = LpPerturbationModels.get_p(LpPerturbationModels.L2)

        self.eps_init = eps_init
        self.gamma = gamma

        super().__init__(
            step_size=1.0,
            y_target=y_target,
            num_steps=num_steps,
            loss_function=CE_LOSS,
            optimizer_cls=OptimizerFactory.create_sgd(lr=1.0),
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
            LpPerturbationModels.L2,
        }

    def _init_epsilons(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.ones(samples.shape[0]).fill_(self.eps_init)

    def _update_epsilons(
        self,
        is_adv: torch.Tensor,
        epsilons: torch.Tensor,
        best_distances: torch.Tensor,
        gamma: float,
        scores: torch.Tensor,
        target: torch.Tensor,
        delta: torch.Tensor,
        adv_found: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            is_adv,
            epsilons * (1 - gamma),
            epsilons * (1 + gamma),
        )

    def _update_gamma(self, i: int) -> float:
        """Update gamma ."""
        return self.gamma
