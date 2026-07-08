"""Gaussian Blur attack implementation (Foolbox-only backend)."""

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


class GaussianBlur(BaseEvasionAttackCreator):
    """Creator for the decision-based Gaussian Blur attack."""

    def __new__(
        cls,
        steps: int = 1000,
        channel_axis: int | None = None,
        max_sigma: float | None = None,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Gaussian Blur attack.

        Decision-based attack that blurs the inputs with a Gaussian filter of
        linearly increasing standard deviation until the input is
        misclassified, minimizing the L2 norm of the resulting perturbation.

        Parameters
        ----------
        steps : int, optional
            Number of sigma values to try in the linear search.
            Default is 1000.
        channel_axis : int | None, optional
            Index of the channel axis. If None, it is inferred from the model.
            Default is None.
        max_sigma : float | None, optional
            Maximum standard deviation of the Gaussian filter. If None, it is
            derived from the input shape. Default is None.
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
            Gaussian Blur attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            steps=steps,
            channel_axis=channel_axis,
            max_sigma=max_sigma,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the Gaussian Blur attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[GaussianBlurFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_gaussian_blur import (
                GaussianBlurFoolbox,
            )

            return GaussianBlurFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
