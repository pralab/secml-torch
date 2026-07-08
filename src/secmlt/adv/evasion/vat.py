"""Virtual Adversarial Training (VAT) attack implementation."""

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


class VAT(BaseEvasionAttackCreator):
    """Creator for the Virtual Adversarial Training attack."""

    def __new__(
        cls,
        epsilon: float,
        steps: int = 1,
        xi: float = 1e-6,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Virtual Adversarial Training attack.

        References
        ----------
        .. [#Miy15] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae,
            Shin Ishii, "Distributional Smoothing with Virtual Adversarial
            Training", https://arxiv.org/abs/1507.00677

        Parameters
        ----------
        epsilon : float
            Maximum L2 perturbation allowed.
        steps : int, optional
            Number of update steps for the approximated second-order
            optimization. Default is 1.
        xi : float, optional
            L2 distance between original image and first adversarial
            proposal. Default is 1e-6.
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
            VAT attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            epsilon=epsilon,
            steps=steps,
            xi=xi,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the VAT attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[VATFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_vat import VATFoolbox

            return VATFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
