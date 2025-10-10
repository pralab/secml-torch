"""Wrapper of the FMN attack implemented in Adversarial Library."""

from __future__ import annotations  # noqa: I001
from functools import partial

from adv_lib.attacks import fmn
from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class FMNAdvLib(BaseAdvLibEvasionAttack):
    """Wrapper of the Adversarial Library implementation of the FMN attack."""

    def __init__(
        self,
        perturbation_model: str,
        num_steps: int,
        max_step_size: float,
        min_step_size: float | None = None,
        gamma: float | None = 0.05,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initialize a FMN attack with the Adversarial Library backend.

        Parameters
        ----------
        perturbation_model : str
            The perturbation model to be used for the attack.
        num_steps : int
            The number of iterations for the attack.
        max_step_size : float
            The attack maximum step size.
        min_step_size : float, optional
            The attack minimum step size. If None, it is set to max_step_size/100.
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
        perturbation_models = {
            LpPerturbationModels.L0: partial(fmn, norm=0),
            LpPerturbationModels.L1: partial(fmn, norm=1),
            LpPerturbationModels.L2: partial(fmn, norm=2),
            LpPerturbationModels.LINF: partial(fmn, norm=float("inf")),
        }

        advlib_attack_func = perturbation_models.get(perturbation_model)
        advlib_attack = partial(
            advlib_attack_func,
            steps=num_steps,
            α_init=max_step_size,
            α_final=min_step_size,
            γ_init=gamma,
        )

        super().__init__(
            advlib_attack=advlib_attack, y_target=y_target, lb=lb, ub=ub, **kwargs
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
