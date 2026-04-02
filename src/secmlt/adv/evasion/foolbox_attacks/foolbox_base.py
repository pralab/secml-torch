"""Generic wrapper for Foolbox evasion attacks."""

from __future__ import annotations  # noqa: I001

from typing import Literal, TYPE_CHECKING

import torch
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.models.pytorch import PyTorchModel
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.trackers.model_tracker import ModelTracker

if TYPE_CHECKING:
    from foolbox.attacks.base import Attack
    from secmlt.models.base_model import BaseModel
    from secmlt.trackers.trackers import Tracker


class BaseFoolboxEvasionAttack(BaseEvasionAttack):
    """Generic wrapper for Foolbox Evasion attacks."""

    def __init__(
        self,
        foolbox_attack: type[Attack],
        epsilon: float = torch.inf,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: list[Tracker] | Tracker | None = None,
    ) -> None:
        """
        Wrap Foolbox attacks.

        Parameters
        ----------
        foolbox_attack : Type[Attack]
            Foolbox attack class to wrap.
        epsilon : float, optional
            Perturbation constraint, by default torch.inf.
        y_target : int | None, optional
            Target label for the attack, None if untargeted, by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        trackers : type[TRACKER_TYPE] | None, optional
            Trackers for the attack, by default None.
        """
        self.foolbox_attack = foolbox_attack
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon
        self.y_target = y_target
        self.trackers = trackers
        super().__init__()

    @classmethod
    def _trackers_allowed(cls) -> Literal[True]:
        return True

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(model, BasePyTorchClassifier):
            msg = "Model type not supported."
            raise NotImplementedError(msg)
        target = None
        if self.y_target is not None:
            target = (torch.zeros_like(labels) + self.y_target).type(labels.dtype)
        tracking_labels = labels if target is None else target
        # Wrap model with ModelTracker if trackers are set
        model_tracker = None
        if self._trackers:
            model_tracker = ModelTracker(model, trackers=self._trackers)
            model_tracker.init_tracking(x_orig=samples, y=tracking_labels)
            model = model_tracker
        device = model._get_device()
        samples = samples.to(device)
        labels = labels.to(device)
        foolbox_model = PyTorchModel(model.model, (self.lb, self.ub), device=device)
        if self.y_target is None:
            criterion = Misclassification(labels)
        else:
            target = target.to(device)
            criterion = TargetedMisclassification(target)

        try:
            _, advx, _ = self.foolbox_attack(
                model=foolbox_model,
                inputs=samples,
                criterion=criterion,
                epsilons=self.epsilon,
            )
        finally:
            if model_tracker is not None:
                model_tracker.end_tracking()
                model_tracker.detach()
        # foolbox deals only with additive perturbations
        delta = advx - samples
        return advx, delta
