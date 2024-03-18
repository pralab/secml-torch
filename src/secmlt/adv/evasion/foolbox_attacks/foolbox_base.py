"""Generic wrapper for Foolbox evasion attacks."""

from typing import Literal

import torch
from foolbox.attacks.base import Attack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.models.pytorch import PyTorchModel
from secmlt.adv.evasion.base_evasion_attack import TRACKER_TYPE, BaseEvasionAttack
from secmlt.models.base_model import BaseModel
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


class BaseFoolboxEvasionAttack(BaseEvasionAttack):
    """Generic wrapper for Foolbox Evasion attacks."""

    def __init__(
        self,
        foolbox_attack: type[Attack],
        epsilon: float = torch.inf,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: type[TRACKER_TYPE] | None = None,
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
            Trackers for the attack (unallowed in Foolbox), by default None.
        """
        self.foolbox_attack = foolbox_attack
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon
        self.y_target = y_target
        self.trackers = trackers
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
        foolbox_model = PyTorchModel(model.model, (self.lb, self.ub), device=device)
        if self.y_target is None:
            criterion = Misclassification(labels)
        else:
            target = (
                torch.zeros_like(labels) + self.y_target
                if self.y_target is not None
                else labels
            ).type(labels.dtype)
            criterion = TargetedMisclassification(target)
        _, advx, _ = self.foolbox_attack(
            model=foolbox_model,
            inputs=samples,
            criterion=criterion,
            epsilons=self.epsilon,
        )
        # foolbox deals only with additive perturbations
        delta = advx - samples
        return advx, delta
