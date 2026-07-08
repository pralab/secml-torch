"""Implementation of the Fast Gradient Sign Method (FGSM) evasion attack."""

from __future__ import annotations  # noqa: I001

import importlib.util

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class FGSM(BaseEvasionAttackCreator):
    """Creator for the Fast Gradient Sign Method (FGSM) attack."""

    def __new__(
        cls,
        perturbation_model: str,
        epsilon: float,
        loss_function: str = "ce",
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the FGSM attack.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model for the attack.
        epsilon : float
            Maximum perturbation allowed.
        loss_function : str, optional
            Loss function to use (advlib backend only), by default "ce".
        y_target : int | None, optional
            Target label for a targeted attack, None for untargeted,
            by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        backend : str, optional
            Backend to use to run the attack, by default Backends.FOOLBOX.
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            available only for native implementation, by default None.

        Returns
        -------
        BaseEvasionAttack
            FGSM attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        implementation.check_perturbation_model_available(perturbation_model)
        if backend == Backends.ADVLIB:
            return implementation(
                perturbation_model=perturbation_model,
                epsilon=epsilon,
                loss_function=loss_function,
                y_target=y_target,
                lb=lb,
                ub=ub,
                trackers=trackers,
                **kwargs,
            )
        return implementation(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the FGSM attack."""
        return [Backends.FOOLBOX, Backends.ADVLIB]

    @staticmethod
    def _get_foolbox_implementation() -> type[FGSMFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_fgsm import FGSMFoolbox

            return FGSMFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_advlib_implementation() -> type[FGSMAdvLib]:  # noqa: F821
        if importlib.util.find_spec("adv_lib", None) is not None:
            from secmlt.adv.evasion.advlib_attacks.advlib_fgsm import FGSMAdvLib

            return FGSMAdvLib
        msg = "adv_lib extra not installed"
        raise ImportError(msg)
