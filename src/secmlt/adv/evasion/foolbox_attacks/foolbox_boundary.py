"""Wrapper of the Boundary Attack implemented in Foolbox."""

from __future__ import annotations

from typing import TYPE_CHECKING

from foolbox.attacks.boundary_attack import BoundaryAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

if TYPE_CHECKING:
    from foolbox.attacks.base import MinimizationAttack


class BoundaryAttackFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Boundary Attack.

    Parameters
    ----------
    init_attack : MinimizationAttack | None, optional
        Attack used to find an initial adversarial example. If None, Foolbox
        falls back to LinearSearchBlendedUniformNoiseAttack. Default is None.
    steps : int, optional
        Number of steps to run. Default is 25000.
    spherical_step : float, optional
        Initial step size for the orthogonal (spherical) step. Default is 0.01.
    source_step : float, optional
        Initial step size for the step toward the target. Default is 0.01.
    source_step_convergance : float, optional
        Convergence threshold for the source step. Default is 1e-7.
    step_adaptation : float, optional
        Factor to adapt the step sizes. Default is 1.5.
    update_stats_every_k : int, optional
        How often to update statistics used for step adaptation. Default is 10.
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
        init_attack: MinimizationAttack | None = None,
        steps: int = 25000,
        spherical_step: float = 0.01,
        source_step: float = 0.01,
        source_step_convergance: float = 1e-7,
        step_adaptation: float = 1.5,
        update_stats_every_k: int = 10,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Boundary Attack with Foolbox backend."""
        foolbox_attack = BoundaryAttack(
            init_attack=init_attack,
            steps=steps,
            spherical_step=spherical_step,
            source_step=source_step,
            source_step_convergance=source_step_convergance,
            step_adaptation=step_adaptation,
            update_stats_every_k=update_stats_every_k,
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
