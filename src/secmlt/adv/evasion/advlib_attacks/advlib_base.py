"""Generic wrapper for Adversarial Library evasion attacks."""

from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Literal

import torch
from secmlt.adv.evasion.base_evasion_attack import TRACKER_TYPE, BaseEvasionAttack

from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from secmlt.models.base_model import BaseModel


class BaseAdvLibEvasionAttack(BaseEvasionAttack):
    """Generic wrapper for Adversarial Library Evasion attacks."""

    def __init__(
        self,
        advlib_attack: Callable[..., torch.Tensor],
        epsilon: float = torch.inf,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: type[TRACKER_TYPE] | None = None,
        **kwargs,
    ) -> None:
        """
        Wrap Adversarial Library attacks.

        Parameters
        ----------
        advlib_attack : Callable[..., torch.Tensor]
            The Adversarial Library attack function to wrap.
            The function returns the adversarial examples.
        epsilon : float, optional
            The perturbation constraint. The default value is
            torch.inf, which means no constraint.
        y_target : int | None, optional
            The target label for the attack. If None, the attack is
            untargeted. The default value is None.
        lb : float, optional
            The lower bound for the perturbation. The default value is 0.0.
        ub : float, optional
            The upper bound for the perturbation. The default value is 1.0.
        trackers : type[TRACKER_TYPE] | None, optional
            Trackers for the attack (unallowed in Adversarial Library), by default None.
        """
        self.advlib_attack = advlib_attack
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon
        self.y_target = y_target
        self.trackers = trackers
        self.kwargs = kwargs
        super().__init__()

    @classmethod
    def _trackers_allowed(cls) -> Literal[False]:
        return False

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(model, BasePytorchClassifier):
            msg = "Model type not supported."
            raise NotImplementedError(msg)
        device = model._get_device()
        samples = samples.to(device)
        if self.y_target is not None:
            targets = torch.ones_like(labels) * self.y_target
        else:
            labels = labels.to(device)
            targets = labels
        if self.epsilon < float(torch.inf):
            self.kwargs.update({"ε": self.epsilon})
        advx = self.advlib_attack(
            model=model,
            inputs=samples,
            labels=targets,
            targeted=(self.y_target is not None),
            **self.kwargs,
        )

        delta = advx - samples
        return advx, delta
