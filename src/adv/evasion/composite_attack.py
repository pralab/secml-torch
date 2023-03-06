from typing import Union, List, Type

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from src.adv.evasion.base_evasion_attack import BaseEvasionAttack
from src.manipulations.manipulation import Manipulation
from src.models.base_model import BaseModel
from src.optimization.constraints import Constraint
from src.optimization.gradient_processing import GradientProcessing
from src.optimization.initializer import Initializer

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
        adversarials = []
        original_labels = []
        multiplier = 1 if self.y_target is not None else -1
        for samples, labels in data_loader:
            target = (
                torch.zeros_like(labels) + self.y_target
                if self.y_target is not None
                else labels
            )

            delta = self.initializer(samples.data)
            delta.requires_grad = True
            optimizer = self.optimizer_cls([delta], lr=self.step_size)
            perturbation_constraints = self.init_perturbation_constraints(samples)
            x_adv = self.manipulation_function(samples, delta)
            for i in range(self.num_steps):
                scores = model.decision_function(x_adv)
                target = target.to(scores.device)
                loss = self.loss_function(scores, target) * multiplier
                optimizer.zero_grad()
                loss.backward()
                gradient = delta.grad
                gradient = self.gradient_processing(gradient)
                delta.grad.data = gradient.data
                optimizer.step()
                for constraint in perturbation_constraints:
                    delta.data = constraint(delta.data)
                x_adv = self.manipulation_function(samples, delta)
                for constraint in self.domain_constraints:
                    x_adv.data = constraint(x_adv.data)

            adversarials.append(x_adv)
            original_labels.append(labels)
            # print('NORM : ', delta.flatten(start_dim=1).norm(p=float('inf')))
            #TODO check best according to custom metric

        adversarials = torch.vstack(adversarials)
        original_labels = torch.hstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        adversarial_loader = DataLoader(
            adversarial_dataset, batch_size=data_loader.batch_size
        )
        return adversarial_loader
