"""Spatial Attack implementation (Foolbox-only backend)."""

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


class SpatialAttack(BaseEvasionAttackCreator):
    """Creator for the Spatial Attack."""

    def __new__(
        cls,
        max_translation: float = 3,
        max_rotation: float = 30,
        num_translations: int = 5,
        num_rotations: int = 5,
        grid_search: bool = True,
        random_steps: int = 100,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Spatial Attack.

        References
        ----------
        .. [#Engs19] Logan Engstrom, Brandon Tran, Dimitris Tsipras,
            Ludwig Schmidt, Aleksander Madry,
            "Exploring the Landscape of Spatial Robustness",
            https://arxiv.org/abs/1712.02779

        Parameters
        ----------
        max_translation : float, optional
            Maximum translation in any direction. Default is 3.
        max_rotation : float, optional
            Maximum rotation in degrees. Default is 30.
        num_translations : int, optional
            Number of translations to try in the grid search (per axis).
            Only used when ``grid_search`` is True. Default is 5.
        num_rotations : int, optional
            Number of rotations to try in the grid search. Only used when
            ``grid_search`` is True. Default is 5.
        grid_search : bool, optional
            If True, perform a grid search over the translation/rotation
            space. If False, perform a random search. Default is True.
        random_steps : int, optional
            Number of random search steps. Only used when ``grid_search``
            is False. Default is 100.
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
            Spatial Attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            max_translation=max_translation,
            max_rotation=max_rotation,
            num_translations=num_translations,
            num_rotations=num_rotations,
            grid_search=grid_search,
            random_steps=random_steps,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the Spatial Attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[SpatialAttackFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_spatial import (
                SpatialAttackFoolbox,
            )

            return SpatialAttackFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
