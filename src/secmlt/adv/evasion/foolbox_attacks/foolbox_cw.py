"""Wrapper of the Carlini-Wagner attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.carlini_wagner import L2CarliniWagnerAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class CWFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Carlini-Wagner (CW) L2 attack.

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
        """Create CW L2 attack with Foolbox backend."""
        foolbox_attack = L2CarliniWagnerAttack(
            binary_search_steps=binary_search_steps,
            steps=num_steps,
            stepsize=step_size,
            confidence=confidence,
            initial_const=initial_const,
            abort_early=abort_early,
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
        return {LpPerturbationModels.L2}
