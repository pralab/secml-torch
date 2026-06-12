"""Additive Noise attack implementation (Foolbox-only backend)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Literal

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class AdditiveNoise(BaseEvasionAttackCreator):
    """Creator for the Additive Noise attack."""

    def __new__(
        cls,
        epsilon: float,
        perturbation_model: str = LpPerturbationModels.L2,
        noise_type: Literal["gaussian", "uniform"] = "gaussian",
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Additive Noise attack.

        The attack samples a random perturbation (Gaussian or uniform) and
        scales it to the requested Lp norm.

        Parameters
        ----------
        epsilon : float
            Maximum size of the additive perturbation, measured with the norm
            induced by ``perturbation_model``.
        perturbation_model : str, optional
            Norm constraint for the attack. Either L2 or Linf. Default is L2.
        noise_type : str, optional
            Distribution used to sample the noise. Either "gaussian" or
            "uniform". Default is "gaussian". Note that "gaussian" is only
            available for the L2 perturbation model.
        y_target : int | None, optional
            Target label for targeted attack. If None, the attack is
            untargeted. Default is None.
        lb : float, optional
            Lower bound for the input domain. Default is 0.0.
        ub : float, optional
            Upper bound for the input domain. Default is 1.0.
        backend : str, optional
            Backend to use. Only Backends.FOOLBOX is supported.
            Default is Backends.FOOLBOX.
        trackers : list[Tracker] | None, optional
            Trackers for monitoring attack metrics, by default None.

        Returns
        -------
        BaseEvasionAttack
            Additive Noise attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            epsilon=epsilon,
            perturbation_model=perturbation_model,
            noise_type=noise_type,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the Additive Noise attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[AdditiveNoiseFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_additive_noise import (
                AdditiveNoiseFoolbox,
            )

            return AdditiveNoiseFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
