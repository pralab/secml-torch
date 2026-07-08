"""Wrapper of the Contrast Reduction attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.contrast import L2ContrastReductionAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class ContrastReductionFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Contrast Reduction attack.

    Reduces the contrast of the input, moving every pixel towards the
    ``target`` value, using an additive perturbation of the given L2 size.

    Parameters
    ----------
    epsilon : float
        Maximum L2 size of the contrast-reducing perturbation.
    target : float, optional
        Value towards which the pixels are moved when reducing the contrast.
        Default is 0.5.
    y_target : int | None, optional
        Target label for targeted attack. If None, the attack is untargeted.
        Default is None.
    lb : float, optional
        Lower bound for the input domain. Default is 0.0.
    ub : float, optional
        Upper bound for the input domain. Default is 1.0.
    """

    def __init__(
        self,
        epsilon: float,
        target: float = 0.5,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Contrast Reduction attack with Foolbox backend."""
        foolbox_attack = L2ContrastReductionAttack(target=target)

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Check the perturbation models implemented for this attack."""
        return {LpPerturbationModels.L2}
