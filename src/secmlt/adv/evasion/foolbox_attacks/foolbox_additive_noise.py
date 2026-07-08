"""Wrapper of the Additive Noise attacks implemented in Foolbox."""

from __future__ import annotations

from typing import ClassVar, Literal

from foolbox.attacks.additive_noise import (
    L2AdditiveGaussianNoiseAttack,
    L2AdditiveUniformNoiseAttack,
    LinfAdditiveUniformNoiseAttack,
)
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class AdditiveNoiseFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Additive Noise attacks.

    The attack samples a random perturbation (Gaussian or uniform) and scales
    it to the requested Lp norm. The supported combinations of perturbation
    model and noise type are:

    * ``L2`` with ``"gaussian"`` (default)
    * ``L2`` with ``"uniform"``
    * ``Linf`` with ``"uniform"``

    Parameters
    ----------
    epsilon : float
        Maximum size of the additive perturbation, measured with the norm
        induced by ``perturbation_model``.
    perturbation_model : str, optional
        Norm constraint for the attack. Either L2 or Linf. Default is L2.
    noise_type : str, optional
        Distribution used to sample the noise. Either "gaussian" or "uniform".
        Default is "gaussian". Note that "gaussian" is only available for the
        L2 perturbation model.
    y_target : int | None, optional
        Target label for targeted attack. If None, the attack is untargeted.
        Default is None.
    lb : float, optional
        Lower bound for the input domain. Default is 0.0.
    ub : float, optional
        Upper bound for the input domain. Default is 1.0.
    """

    _ATTACK_MAP: ClassVar[dict[tuple[str, str], type]] = {
        (LpPerturbationModels.L2, "gaussian"): L2AdditiveGaussianNoiseAttack,
        (LpPerturbationModels.L2, "uniform"): L2AdditiveUniformNoiseAttack,
        (LpPerturbationModels.LINF, "uniform"): LinfAdditiveUniformNoiseAttack,
    }

    def __init__(
        self,
        epsilon: float,
        perturbation_model: str = LpPerturbationModels.L2,
        noise_type: Literal["gaussian", "uniform"] = "gaussian",
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Additive Noise attack with Foolbox backend."""
        foolbox_attack_cls = self._ATTACK_MAP.get((perturbation_model, noise_type))
        if foolbox_attack_cls is None:
            msg = (
                f"Unsupported combination of perturbation model "
                f"'{perturbation_model}' and noise type '{noise_type}' for the "
                f"Additive Noise attack."
            )
            raise NotImplementedError(msg)
        foolbox_attack = foolbox_attack_cls()

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Check the perturbation models implemented for this attack."""
        return {LpPerturbationModels.L2, LpPerturbationModels.LINF}
