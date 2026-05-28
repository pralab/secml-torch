"""Wrapper of the Carlini-Wagner attack implemented in Adversarial Library."""

from __future__ import annotations  # noqa: I001

from functools import partial

from adv_lib.attacks.carlini_wagner import carlini_wagner_l2

from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class CWAdvLib(BaseAdvLibEvasionAttack):
    """Wrapper of the Adversarial Library implementation of the CW L2 attack.

    Parameters
    ----------
        binary_search_steps : int
            Number of binary search steps for the const parameter. Default is 9.
        num_steps : int
            Number of optimization iterations per binary search step. Default is 10000.
        step_size : float
            Learning rate for the Adam optimizer. Default is 0.01.
        confidence : float
            Confidence margin for adversarial examples. Default is 0.0.
        initial_const : float
            Initial value of the regularization constant. Default is 0.001.
        abort_early : bool
            Abort binary search early if no improvement is found. Default is True.
        y_target : int | None, optional
            Target label for the attack. If None, the attack is untargeted.
        lb : float, optional
            Lower bound for the perturbation. Default is 0.0.
        ub : float, optional
            Upper bound for the perturbation. Default is 1.0.
    """

    def __init__(
        self,
        binary_search_steps: int = 9,
        num_steps: int = 10000,
        step_size: float = 0.01,
        confidence: float = 0.0,
        initial_const: float = 0.001,
        abort_early: bool = True,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the Adversarial Library backend for the CW L2 attack."""
        advlib_attack = partial(
            carlini_wagner_l2,
            binary_search_steps=binary_search_steps,
            max_iterations=num_steps,
            learning_rate=step_size,
            confidence=confidence,
            initial_const=initial_const,
            abort_early=abort_early,
        )

        super().__init__(
            advlib_attack=advlib_attack,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Return the perturbation models available for this attack."""
        return {LpPerturbationModels.L2}
