"""Carlini and Wagner (CW) L2 attack implementation."""

from __future__ import annotations  # noqa: I001

import importlib.util

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class CW(BaseEvasionAttackCreator):
    """Creator for the Carlini and Wagner (CW) L2 attack."""

    def __new__(
        cls,
        binary_search_steps: int = 9,
        num_steps: int = 10000,
        step_size: float = 0.01,
        confidence: float = 0.0,
        initial_const: float = 0.001,
        abort_early: bool = True,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the CW L2 attack.

        References
        ----------
        .. [#CW17] Nicholas Carlini and David Wagner, "Towards Evaluating the
            Robustness of Neural Networks", IEEE S&P 2017,
            https://arxiv.org/abs/1608.04644

        Parameters
        ----------
        binary_search_steps : int, optional
            Number of binary search steps for the regularization constant.
            Default is 9.
        num_steps : int, optional
            Number of optimization iterations per binary search step.
            Default is 10000.
        step_size : float, optional
            Learning rate for the Adam optimizer. Default is 0.01.
        confidence : float, optional
            Confidence margin kappa for the adversarial objective. Default is 0.0.
        initial_const : float, optional
            Initial value of the regularization constant c. Default is 0.001.
        abort_early : bool, optional
            Abort binary search early when no improvement is observed.
            Default is True.
        y_target : int | None, optional
            Target label for the attack. If None, the attack is untargeted.
            Default is None.
        lb : float, optional
            Lower bound for the input domain. Default is 0.0.
        ub : float, optional
            Upper bound for the input domain. Default is 1.0.
        backend : str, optional
            Backend to use to run the attack. Default is Backends.FOOLBOX.
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            by default None.

        Returns
        -------
        BaseEvasionAttack
            CW L2 attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        return implementation(
            binary_search_steps=binary_search_steps,
            num_steps=num_steps,
            step_size=step_size,
            confidence=confidence,
            initial_const=initial_const,
            abort_early=abort_early,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the CW attack."""
        return [Backends.FOOLBOX, Backends.ADVLIB]

    @staticmethod
    def _get_foolbox_implementation() -> type[CWFoolbox]:  # noqa: F821
        if importlib.util.find_spec("foolbox", None) is not None:
            from secmlt.adv.evasion.foolbox_attacks.foolbox_cw import CWFoolbox

            return CWFoolbox
        msg = "foolbox extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_advlib_implementation() -> type[CWAdvLib]:  # noqa: F821
        if importlib.util.find_spec("adv_lib", None) is not None:
            from secmlt.adv.evasion.advlib_attacks.advlib_cw import CWAdvLib

            return CWAdvLib
        msg = "adv_lib extra not installed"
        raise ImportError(msg)
