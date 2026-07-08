"""Boundary Attack implementation (Foolbox-only backend)."""

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


class BoundaryAttack(BaseEvasionAttackCreator):
    """Creator for the decision-based Boundary Attack."""

    def __new__(
        cls,
        steps: int = 25000,
        spherical_step: float = 0.01,
        source_step: float = 0.01,
        source_step_convergance: float = 1e-7,
        step_adaptation: float = 1.5,
        update_stats_every_k: int = 10,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Boundary Attack.

        References
        ----------
        .. [#Bren17] Wieland Brendel, Jonas Rauber, Matthias Bethge,
            "Decision-Based Adversarial Attacks: Reliable Attacks Against
            Black-Box Machine Learning Models",
            https://arxiv.org/abs/1712.04248

        Parameters
        ----------
        steps : int, optional
            Number of steps to run. Default is 25000.
        spherical_step : float, optional
            Initial step size for the orthogonal (spherical) step.
            Default is 0.01.
        source_step : float, optional
            Initial step size for the step toward the target. Default is 0.01.
        source_step_convergance : float, optional
            Convergence threshold for the source step. Default is 1e-7.
        step_adaptation : float, optional
            Factor to adapt the step sizes. Default is 1.5.
        update_stats_every_k : int, optional
            How often to update statistics used for step adaptation.
            Default is 10.
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
            Boundary Attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            steps=steps,
            spherical_step=spherical_step,
            source_step=source_step,
            source_step_convergance=source_step_convergance,
            step_adaptation=step_adaptation,
            update_stats_every_k=update_stats_every_k,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the Boundary Attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[BoundaryAttackFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_boundary import (
                BoundaryAttackFoolbox,
            )

            return BoundaryAttackFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
