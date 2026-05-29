"""Wrapper of the Virtual Adversarial Attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.virtual_adversarial_attack import VirtualAdversarialAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class VATFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Virtual Adversarial Attack.

    Parameters
    ----------
    epsilon : float
        Maximum L2 perturbation allowed.
    steps : int, optional
        Number of update steps for the approximated second-order optimization.
        Default is 1.
    xi : float, optional
        L2 distance between original image and first adversarial proposal.
        Default is 1e-6.
    lb : float, optional
        Lower bound for the input domain. Default is 0.0.
    ub : float, optional
        Upper bound for the input domain. Default is 1.0.
    """

    def __init__(
        self,
        epsilon: float,
        steps: int = 1,
        xi: float = 1e-6,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Virtual Adversarial Attack with Foolbox backend."""
        foolbox_attack = VirtualAdversarialAttack(steps=steps, xi=xi)

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=None,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Check the perturbation models implemented for this attack."""
        return {LpPerturbationModels.L2}
