"""DeepFool L2 attack implementation."""

from __future__ import annotations

import importlib.util

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class DeepFool(BaseEvasionAttackCreator):
    """Creator for the DeepFool L2 attack."""

    def __new__(
        cls,
        num_steps: int = 50,
        overshoot: float = 0.02,
        candidates: int | None = 10,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the DeepFool L2 attack.

        References
        ----------
        .. [#Moos15] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
            "DeepFool: a simple and accurate method to fool deep neural
            networks", https://arxiv.org/abs/1511.04599

        Parameters
        ----------
        num_steps : int, optional
            Maximum number of steps to perform. Default is 50.
        overshoot : float, optional
            How much to overshoot the decision boundary. Default is 0.02.
        candidates : int | None, optional
            Limit on the number of the most likely classes to consider
            (Foolbox backend only). Default is 10.
        lb : float, optional
            Lower bound for the input domain. Default is 0.0.
        ub : float, optional
            Upper bound for the input domain. Default is 1.0.
        backend : str, optional
            Backend to use to run the attack. Default is Backends.FOOLBOX.
        trackers : list[Tracker] | None, optional
            Trackers for monitoring attack metrics, by default None.

        Returns
        -------
        BaseEvasionAttack
            DeepFool L2 attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            num_steps=num_steps,
            overshoot=overshoot,
            candidates=candidates,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the DeepFool attack."""
        return [Backends.FOOLBOX, Backends.ADVLIB]

    @staticmethod
    def _get_foolbox_implementation() -> type[DeepFoolFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_deepfool import (
                DeepFoolFoolbox,
            )

            return DeepFoolFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_advlib_implementation() -> type[DeepFoolAdvLib]:  # noqa: F821
        if importlib.util.find_spec("adv_lib", None) is not None:
            from secmlt.adv.evasion.advlib_attacks.advlib_deepfool import (
                DeepFoolAdvLib,
            )

            return DeepFoolAdvLib
        msg = "adv_lib extra not installed"
        raise ImportError(msg)
