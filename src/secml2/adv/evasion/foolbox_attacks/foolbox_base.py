from typing import Optional, Type, Union
from secml2.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secml2.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secml2.models.base_model import BaseModel
from foolbox.attacks.base import Attack
from foolbox.models.pytorch import PyTorchModel
from foolbox.criteria import Misclassification, TargetedMisclassification
from secml2.trackers.tracker import Tracker
import torch


class BaseFoolboxEvasionAttack(BaseEvasionAttack):
    def __init__(
        self,
        foolbox_attack: Attack,
        epsilon: float = torch.inf,
        y_target: Optional[int] = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: Union[Type[Tracker], None] = None,
    ) -> None:
        self.foolbox_attack = foolbox_attack
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon
        self.y_target = y_target
        self.trackers = trackers
        super().__init__()

    def _run(
        self, model: BaseModel, samples: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # TODO get here the correct model if not pytorch
        if not isinstance(model, BasePytorchClassifier):
            raise NotImplementedError("Model type not supported.")
        device = model.get_device()
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
        return advx
