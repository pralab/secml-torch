"""Wrapper of the DeepFool attack implemented in Adversarial Library."""

from __future__ import annotations

from functools import partial

from adv_lib.attacks.deepfool import df
from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class DeepFoolAdvLib(BaseAdvLibEvasionAttack):
    """Wrapper of the Adversarial Library implementation of the DeepFool L2 attack.

    Parameters
    ----------
        num_steps : int
            Maximum number of steps to perform. Default is 100.
        overshoot : float, optional
            Ratio by which to overshoot the decision boundary. Default is 0.02.
        lb : float, optional
            Lower bound for the perturbation. Default is 0.0.
        ub : float, optional
            Upper bound for the perturbation. Default is 1.0.
    """

    def __init__(
        self,
        num_steps: int = 100,
        overshoot: float = 0.02,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the Adversarial Library backend for the DeepFool L2 attack."""
        kwargs.pop("candidates", None)  # foolbox-only parameter
        advlib_attack = partial(
            df,
            steps=num_steps,
            overshoot=overshoot,
            norm=2,
        )

        super().__init__(
            advlib_attack=advlib_attack,
            y_target=None,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Return the perturbation models available for this attack."""
        return {LpPerturbationModels.L2}
