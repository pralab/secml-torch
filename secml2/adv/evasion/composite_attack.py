from typing import Union, List, Type

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from secml2.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secml2.manipulations.manipulation import Manipulation
from secml2.models.base_model import BaseModel
from secml2.optimization.constraints import Constraint
from secml2.optimization.gradient_processing import GradientProcessing
from secml2.optimization.initializer import Initializer

CE_LOSS = "ce_loss"
LOGIT_LOSS = "logits_loss"

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
}

ADAM = "adam"
StochasticGD = "sgd"

OPTIMIZERS = {ADAM: Adam, StochasticGD: SGD}


class CompositeEvasionAttack(BaseEvasionAttack):
    def __init__(
        self,
        y_target: Union[int, None],
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: Union[str, Type[Optimizer]],
        manipulation_function: Manipulation,
        domain_constraints: List[Constraint],
        perturbation_constraints: List[Type[Constraint]],
        initializer: Initializer,
        gradient_processing: GradientProcessing,
    ):
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size

        if isinstance(loss_function, str):
            if loss_function in LOSS_FUNCTIONS:
                self.loss_function = LOSS_FUNCTIONS[loss_function]()
            else:
                raise ValueError(
                    f"{loss_function} not in list of init from string. Use one among {LOSS_FUNCTIONS.values()}"
                )
        else:
            self.loss_function = loss_function

        if isinstance(optimizer_cls, str):
            if optimizer_cls in OPTIMIZERS:
                self.optimizer_cls = OPTIMIZERS[optimizer_cls]
            else:
                raise ValueError(
                    f"{optimizer_cls} not in list of init from string. Use one among {OPTIMIZERS.values()}"
                )
        else:
            self.optimizer_cls = optimizer_cls

        self.manipulation_function = manipulation_function
        self.perturbation_constraints = perturbation_constraints
        self.domain_constraints = domain_constraints
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__()

    def init_perturbation_constraints(self) -> List[Constraint]:
        raise NotImplementedError("Must be implemented accordingly")

    def _run(
        self, model: BaseModel, samples: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        multiplier = 1 if self.y_target is not None else -1
        perturbation_constraints = self.init_perturbation_constraints()

        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)
        delta = self.initializer(samples.data)
        delta.requires_grad = True
        optimizer = self.optimizer_cls([delta], lr=self.step_size)
        x_adv = self.manipulation_function(samples, delta)
        for i in range(self.num_steps):
            scores = model.decision_function(x_adv)
            target = target.to(scores.device)
            loss = self.loss_function(scores, target)
            loss = loss * multiplier
            optimizer.zero_grad()
            loss.backward()
            delta.grad.data = self.gradient_processing(delta.grad.data)
            optimizer.step()
            for constraint in perturbation_constraints:
                delta.data = constraint(delta.data)
            x_adv.data = self.manipulation_function(samples.data, delta.data)
            for constraint in self.domain_constraints:
                x_adv.data = constraint(x_adv.data)
            delta.data = self.manipulation_function.invert(samples.data, x_adv.data)
        return x_adv
