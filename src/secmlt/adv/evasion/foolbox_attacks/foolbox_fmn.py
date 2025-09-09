"""Wrapper of the FMN attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.fast_minimum_norm import (
    L0FMNAttack,
    L1FMNAttack,
    L2FMNAttack,
    LInfFMNAttack,
)
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class FMNFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the FMN attack."""

    def __init__(
        self,
        perturbation_model: str,
        num_steps: int,
        max_step_size: float,
        min_step_size: float | None = None,
        gamma: float = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Create FMN attack with Foolbox backend.

        Parameters
        ----------
        perturbation_model : str
            The perturbation model to be used for the attack.
        num_steps : int
            The number of iterations for the attack.
        max_step_size : float
            The attack step size.
        min_step_size : float, optional
            The number of attack restarts. The default value is 1.
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
        perturbation_models = {
            LpPerturbationModels.L0: L0FMNAttack,
            LpPerturbationModels.L1: L1FMNAttack,
            LpPerturbationModels.L2: L2FMNAttack,
            LpPerturbationModels.LINF: LInfFMNAttack,
        }
        foolbox_attack_cls = perturbation_models.get(perturbation_model)

        foolbox_attack = foolbox_attack_cls(
            max_stepsize=max_step_size,
            min_stepsize=min_step_size,
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
            LpPerturbationModels.L0,
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }
