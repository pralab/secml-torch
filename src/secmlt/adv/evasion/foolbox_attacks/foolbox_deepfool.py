"""Wrapper of the DeepFool attack implemented in Foolbox."""

from __future__ import annotations

from typing import Literal

from foolbox.attacks.deepfool import L2DeepFoolAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class DeepFoolFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the DeepFool L2 attack.

    Parameters
    ----------
        num_steps : int
            Maximum number of steps to perform. Default is 50.
        candidates : int | None, optional
            Limit on the number of the most likely classes to consider.
            Default is 10.
        overshoot : float, optional
            How much to overshoot the boundary. Default is 0.02.
        loss : str, optional
            Loss function to use inside the update step ('logits' or
            'crossentropy'). Default is 'logits'.
        lb : float, optional
            Lower bound for the perturbation. Default is 0.0.
        ub : float, optional
            Upper bound for the perturbation. Default is 1.0.
    """

    def __init__(
        self,
        num_steps: int = 50,
        candidates: int | None = 10,
        overshoot: float = 0.02,
        loss: Literal["logits", "crossentropy"] = "logits",
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create DeepFool L2 attack with Foolbox backend."""
        foolbox_attack = L2DeepFoolAttack(
            steps=num_steps,
            candidates=candidates,
            overshoot=overshoot,
            loss=loss,
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=None,
            y_target=None,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Check the perturbation models implemented for this attack."""
        return {LpPerturbationModels.L2}
