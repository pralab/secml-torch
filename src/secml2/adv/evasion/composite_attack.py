from typing import Union, List, Type

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from functools import partial
from secml2.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secml2.manipulations.manipulation import Manipulation
from secml2.models.base_model import BaseModel
from secml2.optimization.constraints import Constraint
from secml2.optimization.gradient_processing import GradientProcessing
from secml2.optimization.initializer import Initializer
from secml2.optimization.optimizer_factory import OptimizerFactory
from secml2.trackers.tracker import Tracker

CE_LOSS = "ce_loss"
LOGIT_LOSS = "logits_loss"

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
}


class CompositeEvasionAttack(BaseEvasionAttack):
    def __init__(
        self,
        y_target: Union[int, None],
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: Union[str, Type[partial[Optimizer]]],
        manipulation_function: Manipulation,
        domain_constraints: List[Constraint],
        perturbation_constraints: List[Type[Constraint]],
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: list[Tracker] = None,
    ) -> None:
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size
        self._trackers = trackers
        if isinstance(loss_function, str):
            if loss_function in LOSS_FUNCTIONS:
                self.loss_function = LOSS_FUNCTIONS[loss_function](reduction='none')
            else:
                raise ValueError(
                    f"{loss_function} not in list of init from string. Use one among {LOSS_FUNCTIONS.values()}"
                )
        else:
            self.loss_function = loss_function

        if isinstance(optimizer_cls, str):
            optimizer_cls = OptimizerFactory.create_from_name(
                optimizer_cls, lr=step_size
            )

        self.optimizer_cls = optimizer_cls

        self.manipulation_function = manipulation_function
        self.perturbation_constraints = perturbation_constraints
        self.domain_constraints = domain_constraints
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__()

    @BaseEvasionAttack.trackers.setter
    def trackers(self, trackers: Union[List[Tracker], None] = None) -> None:
        self._trackers = trackers

    def init_perturbation_constraints(self) -> List[Constraint]:
        raise NotImplementedError("Must be implemented accordingly")

    def create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
        return self.optimizer_cls([delta], **kwargs)

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        **optim_kwargs,
    ) -> torch.Tensor:
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)
        delta = self.initializer(samples.data)
        delta.requires_grad = True

        optimizer = self.create_optimizer(delta, **optim_kwargs)
        x_adv, delta = self.manipulation_function(samples, delta)

        for i in range(self.num_steps):
            scores = model.decision_function(x_adv)
            target = target.to(scores.device)
            losses = self.loss_function(scores, target)
            loss = losses.sum() * multiplier
            optimizer.zero_grad()
            loss.backward()
            delta.grad.data = self.gradient_processing(delta.grad.data)
            optimizer.step()
            for constraint in self.perturbation_constraints:
                delta.data = constraint(delta.data)
            x_adv.data, delta.data = self.manipulation_function(
                samples.data, delta.data
            )
            for constraint in self.domain_constraints:
                x_adv.data = constraint(x_adv.data)
            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(i, losses.detach(), scores.detach(), delta.detach())
        return x_adv
