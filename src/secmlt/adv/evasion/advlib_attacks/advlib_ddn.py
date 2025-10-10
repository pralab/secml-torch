"""Wrapper of the DDN attack implemented in Adversarial Library."""

from __future__ import annotations  # noqa: I001

from functools import partial

from adv_lib.attacks.decoupled_direction_norm import ddn

from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class DDNAdvLib(BaseAdvLibEvasionAttack):
    """Wrapper of the Adversarial Library implementation of the DDN attack.

    Parameters
    ----------
        num_steps : int
            The number of iterations for the attack.
        eps_init: float, optional
            Initial L2 norm of the perturbation. The default value is 8/255.
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

    def __init__(
        self,
        num_steps: int,
        eps_init: float,
        gamma: float,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the Adversarial Library backend for the DDN attack."""
        advlib_attack_func = ddn
        advlib_attack = partial(
            advlib_attack_func,
            steps=num_steps,
            init_norm=eps_init,
            γ=gamma,
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
