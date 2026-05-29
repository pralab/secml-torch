"""HopSkipJump attack implementation (Foolbox-only backend)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Literal

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

if TYPE_CHECKING:
    from foolbox.attacks.base import MinimizationAttack
    from secmlt.trackers.trackers import Tracker


class HopSkipJump(BaseEvasionAttackCreator):
    """Creator for the decision-based HopSkipJump Attack."""

    def __new__(
        cls,
        perturbation_model: str = LpPerturbationModels.L2,
        init_attack: MinimizationAttack | None = None,
        steps: int = 64,
        initial_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        stepsize_search: Literal[
            "geometric_progression", "grid_search"
        ] = "geometric_progression",
        gamma: float = 1.0,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the HopSkipJump Attack.

        References
        ----------
        .. [#Chen19] Jianbo Chen, Michael I. Jordan, Martin J. Wainwright,
            "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
            https://arxiv.org/abs/1904.02144

        Parameters
        ----------
        perturbation_model : str, optional
            Norm constraint for the attack. Either L2 or Linf.
            Default is L2.
        init_attack : MinimizationAttack | None, optional
            Attack used to find an initial adversarial example. If None,
            Foolbox falls back to LinearSearchBlendedUniformNoiseAttack.
            Default is None.
        steps : int, optional
            Number of steps. Default is 64.
        initial_gradient_eval_steps : int, optional
            Initial number of gradient evaluations per step. Default is 100.
        max_gradient_eval_steps : int, optional
            Maximum number of gradient evaluations per step. Default is 10000.
        stepsize_search : str, optional
            Strategy for step-size search. Either "geometric_progression" or
            "grid_search". Default is "geometric_progression".
        gamma : float, optional
            Factor controlling the step-size growth. Default is 1.0.
        y_target : int | None, optional
            Target label for targeted attack. If None, the attack is
            untargeted. Default is None.
        lb : float, optional
            Lower bound for the input domain. Default is 0.0.
        ub : float, optional
            Upper bound for the input domain. Default is 1.0.
        backend : str, optional
            Backend to use. Only Backends.FOOLBOX is supported.
            Default is Backends.FOOLBOX.
        trackers : list[Tracker] | None, optional
            Trackers for monitoring attack metrics, by default None.

        Returns
        -------
        BaseEvasionAttack
            HopSkipJump attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            perturbation_model=perturbation_model,
            init_attack=init_attack,
            steps=steps,
            initial_gradient_eval_steps=initial_gradient_eval_steps,
            max_gradient_eval_steps=max_gradient_eval_steps,
            stepsize_search=stepsize_search,
            gamma=gamma,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the HopSkipJump attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[HopSkipJumpFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_hopskipjump import (
                HopSkipJumpFoolbox,
            )

            return HopSkipJumpFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
