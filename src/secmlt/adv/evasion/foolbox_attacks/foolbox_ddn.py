"""Wrapper of the DDN attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.ddn import DDNAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class DDNFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the DDN attack."""

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
        """Create DDN attack with Foolbox backend."""
        type(self).check_perturbation_model_available(perturbation_model)

        foolbox_attack = DDNAttack(
            init_epsilon=init_epsilon,
            steps=num_steps,
            gamma=gamma,
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
