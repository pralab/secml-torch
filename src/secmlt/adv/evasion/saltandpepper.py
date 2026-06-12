"""Salt & Pepper Noise attack implementation (Foolbox-only backend)."""

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


class SaltAndPepperNoise(BaseEvasionAttackCreator):
    """Creator for the decision-based Salt & Pepper Noise attack."""

    def __new__(
        cls,
        steps: int = 1000,
        across_channels: bool = True,
        channel_axis: int | None = None,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the Salt & Pepper Noise attack.

        Decision-based attack that increasingly replaces pixels with salt
        (maximum) and pepper (minimum) values until the input is misclassified,
        minimizing the L2 norm of the resulting perturbation.

        Parameters
        ----------
        steps : int, optional
            Number of steps to run. Default is 1000.
        across_channels : bool, optional
            If True, the same salt/pepper mask is applied across all channels
            of a pixel. If False, channels are perturbed independently.
            Default is True.
        channel_axis : int | None, optional
            Index of the channel axis. If None, it is inferred from the model.
            Default is None.
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
            Salt & Pepper Noise attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            steps=steps,
            across_channels=across_channels,
            channel_axis=channel_axis,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the Salt & Pepper Noise attack."""
        return [Backends.FOOLBOX]

    @staticmethod
    def _get_foolbox_implementation() -> type[SaltAndPepperNoiseFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_saltandpepper import (
                SaltAndPepperNoiseFoolbox,
            )

            return SaltAndPepperNoiseFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)
