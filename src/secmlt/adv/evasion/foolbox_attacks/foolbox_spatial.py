"""Wrapper of the Spatial Attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.spatial_attack import SpatialAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack


class SpatialAttackFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Spatial Attack.

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
        If True, perform a grid search over the translation/rotation space.
        If False, perform a random search. Default is True.
    random_steps : int, optional
        Number of random search steps. Only used when ``grid_search`` is
        False. Default is 100.
    y_target : int | None, optional
        Target label for targeted attack. If None, the attack is untargeted.
        Default is None.
    lb : float, optional
        Lower bound for the input domain. Default is 0.0.
    ub : float, optional
        Upper bound for the input domain. Default is 1.0.
    """

    def __init__(
        self,
        max_translation: float = 3,
        max_rotation: float = 30,
        num_translations: int = 5,
        num_rotations: int = 5,
        grid_search: bool = True,
        random_steps: int = 100,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Spatial Attack with Foolbox backend."""
        foolbox_attack = SpatialAttack(
            max_translation=max_translation,
            max_rotation=max_rotation,
            num_translations=num_translations,
            num_rotations=num_rotations,
            grid_search=grid_search,
            random_steps=random_steps,
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=None,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Check the perturbation models implemented for this attack.

        The Spatial Attack applies rotations and translations rather than
        additive Lp-bounded perturbations, so no Lp perturbation model
        applies.
        """
        return set()
