"""Wrapper of the HopSkipJump Attack implemented in Foolbox."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from foolbox.attacks.hop_skip_jump import HopSkipJumpAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

if TYPE_CHECKING:
    from foolbox.attacks.base import MinimizationAttack


class HopSkipJumpFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the HopSkipJump Attack.

    Parameters
    ----------
    perturbation_model : str, optional
        Norm constraint for the attack. Either L2 or Linf.
        Default is L2.
    init_attack : MinimizationAttack | None, optional
        Attack used to find an initial adversarial example. If None, Foolbox
        falls back to LinearSearchBlendedUniformNoiseAttack. Default is None.
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
        Target label for targeted attack. If None, the attack is untargeted.
        Default is None.
    lb : float, optional
        Lower bound for the input domain. Default is 0.0.
    ub : float, optional
        Upper bound for the input domain. Default is 1.0.
    """

    _CONSTRAINT_MAP: ClassVar[dict[str, str]] = {
        LpPerturbationModels.L2: "l2",
        LpPerturbationModels.LINF: "linf",
    }

    def __init__(
        self,
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
        **kwargs,
    ) -> None:
        """Create HopSkipJump Attack with Foolbox backend."""
        constraint = self._CONSTRAINT_MAP[perturbation_model]
        foolbox_attack = HopSkipJumpAttack(
            init_attack=init_attack,
            steps=steps,
            initial_gradient_eval_steps=initial_gradient_eval_steps,
            max_gradient_eval_steps=max_gradient_eval_steps,
            stepsize_search=stepsize_search,
            gamma=gamma,
            constraint=constraint,
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=None,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Check the perturbation models implemented for this attack."""
        return {LpPerturbationModels.L2, LpPerturbationModels.LINF}
