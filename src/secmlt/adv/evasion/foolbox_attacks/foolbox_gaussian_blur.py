"""Wrapper of the Gaussian Blur attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.blur import GaussianBlurAttack
from foolbox.distances import l2
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class GaussianBlurFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Gaussian Blur attack.

    Decision-based attack that blurs the inputs with a Gaussian filter of
    linearly increasing standard deviation until the input is misclassified,
    minimizing the L2 norm of the resulting perturbation.

    Parameters
    ----------
    steps : int, optional
        Number of sigma values to try in the linear search. Default is 1000.
    channel_axis : int | None, optional
        Index of the channel axis. If None, it is inferred from the model.
        Default is None.
    max_sigma : float | None, optional
        Maximum standard deviation of the Gaussian filter. If None, it is
        derived from the input shape. Default is None.
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
        steps: int = 1000,
        channel_axis: int | None = None,
        max_sigma: float | None = None,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Gaussian Blur attack with Foolbox backend."""
        foolbox_attack = GaussianBlurAttack(
            distance=l2,
            steps=steps,
            channel_axis=channel_axis,
            max_sigma=max_sigma,
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
        """Check the perturbation models implemented for this attack."""
        return {LpPerturbationModels.L2}
