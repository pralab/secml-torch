from typing import Union, List, Type

import torch.nn
from torch.utils.data import DataLoader

from src.adv.evasion.base_evasion_attack import BaseEvasionAttack
from src.models.base_model import BaseModel


class Manipulation:
    def __call__(self, x, delta):
        raise NotImplementedError("Abstract manipulation.")


class Constraint:
    def __call__(self, x, delta, epsilon):
        ...


class L2Constraint(Constraint):
    def __init__(self, center, radius):


    def project(self, x):
        return torch.nn.functional.Normalize(x, p=2) * self.radius


class LpConstraint(Constraint):
    def __init__(self, center, radius, p):
        self.p
        self.center = center
        self.radius = radius

    def _project(self, x):
        ...

    def __call__(self, x):
        x = x + self.center
        norm = torch.norm(x, p=p)
        if norm > self.radius:
            delta = self.project(x)
        return delta



class Initializer:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class GradientProcessing:
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        ...



class CompositeEvasionAttack(BaseEvasionAttack):

    def __init__(self, y_target: Union[int, None], num_steps: int, step_size: float,
                 loss_function: Union[str, torch.nn.Module], optimizer_cls: Union[str, Type[torch.nn.Module]],
                 manipulation_function: Manipulation, domain_constraints: List[Constraint], perturbation_constraints: List[Constraint], initializer: Initializer,
                 gradient_processing: GradientProcessing):
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_function = loss_function
        self.optimizer_cls = optimizer_cls
        self.manipulation_function = manipulation_function
        self.perturbation_constraints = perturbation_constraints
        self.domain_constraints = domain_constraints
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        # TODO: create loss function if str
        # TODO: create optimizer if str

        super().__init__()

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        for samples, labels in data_loader:
            target = torch.zeros_like(labels) + self.y_target if self.y_target is not None else labels
            multiplier = 1 if self.y_target is not None else -1
            delta = self.initializer(samples)
            optimizer = self.optimizer_cls([delta])
            x_adv = self.manipulation_function(samples, delta)
            for i in range(self.num_steps):
                loss = self.loss_function(x_adv, target) * multiplier
                loss.backward()
                gradient = delta.grad
                gradient = self.gradient_processing(gradient)
                delta.grad.data = gradient.data
                optimizer.step()
                for constraint in self.perturbation_constraints:
                    delta = constraint(delta)

                x_adv = self.manipulation_function(samples, delta)
                for constraint in self.domain_constraints:
                    x_adv = constraint(x_adv)
