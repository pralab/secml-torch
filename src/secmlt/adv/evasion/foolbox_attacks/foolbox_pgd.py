"""Wrapper of the PGD attack implemented in Foolbox."""

from foolbox.attacks.projected_gradient_descent import (
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
)
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class PGDFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the PGD attack."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Create PGD attack with Foolbox backend.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model for the attack.
        epsilon : float
            Maximum perturbation allowed.
        num_steps : int
            Number of iterations for the attack.
        step_size : float
            Attack step size.
        random_start : bool
            True for randomly initializing the perturbation.
        y_target : int | None, optional
            Target label for the attack, None for untargeted, by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        """
        perturbation_models = {
            LpPerturbationModels.L1: L1ProjectedGradientDescentAttack,
            LpPerturbationModels.L2: L2ProjectedGradientDescentAttack,
            LpPerturbationModels.LINF: LinfProjectedGradientDescentAttack,
        }
        foolbox_attack_cls = perturbation_models.get(perturbation_model)

        foolbox_attack = foolbox_attack_cls(
            abs_stepsize=step_size,
            steps=num_steps,
            random_start=random_start,
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
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
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }
