import math
from typing import Union, List, Type

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from src.adv.evasion.base_evasion_attack import BaseEvasionAttack
from src.adv.evasion.perturbation_models import PerturbationModels
from src.models.base_model import BaseModel
from src.optimization.constraints import Constraint


class Manipulation:
    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract manipulation.")


class AdditiveManipulation(Manipulation):
    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_adv = x + delta
        return x_adv


class Initializer:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        init = torch.zeros_like(x)
        return init


class RandomLpInitializer(Initializer):
    def __init__(self, center, radius, p):
        raise NotImplementedError("Not yet implemented")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented")


class GradientProcessing:
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        ...


class GradientNormalizerProcessing(GradientProcessing):
    def __init__(self, perturbation_model: str = PerturbationModels.L2):
        perturbations_models = {
            PerturbationModels.L1: 1,
            PerturbationModels.L2: 2,
            PerturbationModels.LINF: float('inf')
        }
        if perturbation_model not in perturbations_models:
            raise ValueError(
                f"{perturbation_model} not included in normalizers. Available: {perturbations_models.values()}")
        self.p = perturbations_models[perturbation_model]

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        original_shape = grad.shape
        flat_shape = (grad.shape[0], math.prod(grad.shape[1:]))
        grad = grad.view(flat_shape)
        norm = torch.linalg.norm(grad, ord=self.p, dim=1)
        norm[norm == 0] = 1  # TODO manage zero gradient here
        normalized_grad = torch.div(grad, norm.view(-1, 1))
        normalized_grad = normalized_grad.view(original_shape)
        return normalized_grad


CE_LOSS = 'ce_loss'
LOGITS_LOSS = 'logits_loss'

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
}

ADAM = 'adam'
StochasticGD = 'sgd'

OPTIMIZERS = {
    ADAM: Adam,
    StochasticGD: SGD
}


class CompositeEvasionAttack(BaseEvasionAttack):
    def __init__(
            self,
            y_target: Union[int, None],
            num_steps: int,
            step_size: float,
            loss_function: Union[str, torch.nn.Module],
            optimizer_cls: Union[str, Type[torch.nn.Module]],
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
                    f"{loss_function} not in list of init from string. Use one among {LOSS_FUNCTIONS.values()}")
        else:
            self.loss_function = loss_function

        if isinstance(optimizer_cls, str):
            if optimizer_cls in OPTIMIZERS:
                self.optimizer_cls = OPTIMIZERS[optimizer_cls]
            else:
                raise ValueError(
                    f"{optimizer_cls} not in list of init from string. Use one among {OPTIMIZERS.values()}")
        else:
            self.optimizer_cls = optimizer_cls

        self.manipulation_function = manipulation_function
        self.perturbation_constraints = perturbation_constraints
        self.domain_constraints = domain_constraints
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__()

    def init_perturbation_constraints(self, center: torch.Tensor) -> List[Constraint]:
        raise NotImplementedError("Must be implemented accordingly")

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        for samples, labels in data_loader:
            target = (
                torch.zeros_like(labels) + self.y_target
                if self.y_target is not None
                else labels
            )
            multiplier = 1 if self.y_target is not None else -1
            delta = self.initializer(samples)
            delta.requires_grad = True
            optimizer = self.optimizer_cls([delta], lr=self.step_size)
            perturbation_constraints = self.init_perturbation_constraints(samples)
            x_adv = self.manipulation_function(samples, delta)
            for i in range(self.num_steps):
                optimizer.zero_grad()
                scores = model.decision_function(x_adv)
                target = target.to(scores.device)
                loss = self.loss_function(scores, target) * multiplier
                loss.backward()
                gradient = delta.grad
                gradient = self.gradient_processing(gradient)
                delta.grad.data = gradient.data
                optimizer.step()
                for constraint in perturbation_constraints:
                    delta = constraint(delta)

                x_adv = self.manipulation_function(samples, delta)
                for constraint in self.domain_constraints:
                    x_adv = constraint(x_adv)
