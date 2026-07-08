"""Contrast Reduction attack implementation (Foolbox-only backend)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class ContrastReduction(BaseEvasionAttackCreator):
    """Creator for the Contrast Reduction attack."""

    def __new__(
        cls,
        epsilon: float,
        target: float = 0.5,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Contrast Reduction attack.

        Reduces the contrast of the input, moving every pixel towards the
        ``target`` value, using an additive perturbation of the given L2 size.

        Parameters
        ----------
        epsilon : float
            Maximum L2 size of the contrast-reducing perturbation.
        target : float, optional
            Value towards which the pixels are moved when reducing the
            contrast. Default is 0.5.
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
            Contrast Reduction attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            epsilon=epsilon,
            target=target,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the Contrast Reduction attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[ContrastReductionFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_contrast_reduction import (
                ContrastReductionFoolbox,
            )

            return ContrastReductionFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
