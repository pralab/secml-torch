"""Wrapper of the PGD attack implemented in Adversarial Library."""

from functools import partial

from adv_lib.attacks import pgd_linf
from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class PGDAdvLib(BaseAdvLibEvasionAttack):
    """Wrapper of the Adversarial Library implementation of the PGD attack."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        random_start: bool,
        step_size: float,
        restarts: int = 1,
        loss_function: str = "ce",
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initialize a PGD attack with the Adversarial Library backend.

        Parameters
        ----------
        perturbation_model : str
            The perturbation model to be used for the attack.
        epsilon : float
            The maximum perturbation allowed.
        num_steps : int
            The number of iterations for the attack.
        random_start : bool
            If True, the perturbation will be randomly initialized.
        step_size : float
            The attack step size.
        restarts : int, optional
            The number of attack restarts. The default value is 1.
        loss_function : str, optional
            The loss function to be used for the attack. The default value is "ce".
        y_target : int | None, optional
            The target label for the attack. If None, the attack is
            untargeted. The default value is None.
        lb : float, optional
            The lower bound for the perturbation. The default value is 0.0.
        ub : float, optional

        Raises
        ------
        ValueError
            If the provided `loss_function` is not supported by the PGD attack
            using the Adversarial Library backend.
        """
        perturbation_models = {
            LpPerturbationModels.LINF: pgd_linf,
        }
        losses = ["ce", "dl", "dlr"]
        if isinstance(loss_function, str):
            if loss_function not in losses:
                msg = f"PGD AdvLib supports only these losses: {losses}"
                raise ValueError(msg)
        else:
            loss_function = losses[0]

        advlib_attack_func = perturbation_models.get(perturbation_model)
        advlib_attack = partial(
            advlib_attack_func,
            steps=num_steps,
            random_init=random_start,
            restarts=restarts,
            loss_function=loss_function,
            absolute_step_size=step_size,
        )

        super().__init__(
            advlib_attack=advlib_attack,
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
            LpPerturbationModels.LINF,
        }
