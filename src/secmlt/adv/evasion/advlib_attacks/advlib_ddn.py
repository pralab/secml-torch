"""Wrapper of the DDN attack implemented in Adversarial Library."""

from __future__ import annotations  # noqa: I001

from functools import partial

from adv_lib.attacks.decoupled_direction_norm import ddn

from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class DDNAdvLib(BaseAdvLibEvasionAttack):
    """Wrapper of the Adversarial Library implementation of the DDN attack."""

    def __init__(
        self,
        perturbation_model: str,
        num_steps: int,
        init_epsilon: float,
        gamma: float,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the Adversarial Library backend for the DDN attack."""
        type(self).check_perturbation_model_available(perturbation_model)

        perturbation_models = {
            LpPerturbationModels.L2: ddn,
        }

        advlib_attack_func = perturbation_models.get(perturbation_model)
        advlib_attack = partial(
            advlib_attack_func,
            steps=num_steps,
            init_norm=init_epsilon,
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
