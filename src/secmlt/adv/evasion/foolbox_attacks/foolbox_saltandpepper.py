"""Wrapper of the Salt & Pepper Noise attack implemented in Foolbox."""

from __future__ import annotations

from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class SaltAndPepperNoiseFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Salt & Pepper Noise attack.

    Decision-based attack that increasingly replaces pixels with salt (maximum)
    and pepper (minimum) values until the input is misclassified, minimizing
    the L2 norm of the resulting perturbation.

    Parameters
    ----------
    steps : int, optional
        Number of steps to run. Default is 1000.
    across_channels : bool, optional
        If True, the same salt/pepper mask is applied across all channels of a
        pixel. If False, channels are perturbed independently. Default is True.
    channel_axis : int | None, optional
        Index of the channel axis. If None, it is inferred from the model.
        Default is None.
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
        across_channels: bool = True,
        channel_axis: int | None = None,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """Create Salt & Pepper Noise attack with Foolbox backend."""
        foolbox_attack = SaltAndPepperNoiseAttack(
            steps=steps,
            across_channels=across_channels,
            channel_axis=channel_axis,
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
