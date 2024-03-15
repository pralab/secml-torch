from typing import Union, List, Type
from secmlt.adv.evasion.perturbation_models import PerturbationModels
from secmlt.utils.tensor_utils import atleast_kd

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from functools import partial
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.manipulations.manipulation import Manipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import Constraint
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.trackers.trackers import Tracker

CE_LOSS = "ce_loss"
LOGIT_LOSS = "logit_loss"

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
}


class ModularEvasionAttackFixedEps(BaseEvasionAttack):
    def __init__(
        self,
        y_target: Union[int, None],
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: Union[str, Type[partial[Optimizer]]],
        manipulation_function: Manipulation,
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: Union[List[Type[Tracker]], Type[Tracker]] = None,
    ) -> None:
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size
        self.trackers = trackers
        if isinstance(loss_function, str):
            if loss_function in LOSS_FUNCTIONS:
                self.loss_function = LOSS_FUNCTIONS[loss_function](reduction="none")
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
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__()

    @classmethod
    def get_perturbation_models(self):
        return {PerturbationModels.L2, PerturbationModels.LINF}

    @classmethod
    def trackers_allowed(cls):
        return True

    def init_perturbation_constraints(self) -> List[Constraint]:
        raise NotImplementedError("Must be implemented accordingly")

    def create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
        return self.optimizer_cls([delta], **kwargs)

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
        **optim_kwargs,
    ) -> torch.Tensor:
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)

        if init_deltas is not None:
            delta = init_deltas.data
        elif isinstance(self.initializer, BaseEvasionAttack):
            _, delta = self.initializer._run(model, samples, target)
        else:
            delta = self.initializer(samples.data)
        delta.requires_grad = True

        optimizer = self.create_optimizer(delta, **optim_kwargs)
        x_adv, delta = self.manipulation_function(samples, delta)
        x_adv.data, delta.data = self.manipulation_function(samples.data, delta.data)
        best_losses = torch.zeros(samples.shape[0]).fill_(torch.inf)
        best_delta = torch.zeros_like(samples)

        for i in range(self.num_steps):
            scores = model.decision_function(x_adv)
            target = target.to(scores.device)
            losses = self.loss_function(scores, target) * multiplier
            loss = losses.sum()
            optimizer.zero_grad()
            loss.backward()
            grad_before_processing = delta.grad.data
            delta.grad.data = self.gradient_processing(delta.grad.data)
            optimizer.step()
            x_adv.data, delta.data = self.manipulation_function(
                samples.data, delta.data
            )
            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(
                        i,
                        losses.detach().cpu().data,
                        scores.detach().cpu().data,
                        x_adv.detach().cpu().data,
                        delta.detach().cpu().data,
                        grad_before_processing.detach().cpu().data,
                    )

            # keep perturbation with highest loss
            best_delta.data = torch.where(
                atleast_kd(losses < best_losses, len(samples.shape)),
                delta.data,
                best_delta.data,
            )
            best_losses.data = torch.where(
                losses < best_losses, losses.data, best_losses.data
            )
        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta
