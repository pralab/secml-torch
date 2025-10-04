"""Wrapper of the DDN attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.ddn import DDNAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class DDNFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the DDN attack."""

    def __init__(
        self,
        num_steps: int,
        eps_init: float = 1.0,
        gamma: float = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Create DDN attack with Foolbox backend.

        Parameters
        ----------
        num_steps : int
            The number of iterations for the attack.
        eps_init : float
            The initial L2 norm of the perturbation. Default is 8/255.
            The default value is None.
        gamma: float, optional
            Step size for modifying the eps-ball. Will decay with cosine annealing.
        y_target : int | None, optional
            The target label for the attack. If None, the attack is
            untargeted. The default value is None.
        lb : float, optional
            The lower bound for the perturbation. The default value is 0.0.
        ub : float, optional
            The upper bound for the perturbation. The default value is 1.0.
        """
        foolbox_attack_cls = DDNAttack

        foolbox_attack = foolbox_attack_cls(
            init_epsilon=eps_init,
            gamma=gamma,
            steps=num_steps,
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
        """
        Check the perturbation models implemented for this attack.

        Returns
        -------
        set[str]
            The list of perturbation models implemented for this attack.
        """
        return {
            LpPerturbationModels.L2,
        }
