from typing import Optional
from src.adv.evasion.base_evasion_attack import BaseEvasionAttack
from foolbox.attacks.base import Attack
from torch.utils.data import DataLoader
from src.models.base_model import BaseModel
from foolbox.models.pytorch import PyTorchModel
from foolbox.criteria import Misclassification, TargetedMisclassification
import torch
from torch.utils.data import TensorDataset


class BaseFoolboxEvasionAttack(BaseEvasionAttack):
    def __init__(
        self,
        foolbox_attack: Attack,
        epsilon: float = torch.inf,
        y_target: Optional[int] = None,
        lb: float = 0.0,
        ub: float = 1.0,
    ) -> None:
        self.foolbox_attack = foolbox_attack
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon
        self.y_target = y_target
        super().__init__()

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        model = self.get_model(model)
        device = model.get_device()
        foolbox_model = PyTorchModel(model.model, (self.lb, self.ub), device=device)
        adversarials = []
        original_labels = []
        for samples, labels in data_loader:
            samples, labels = samples.to(device), labels.to(device)
            if self.y_target is None:
                criterion = Misclassification(labels)
            else:
                criterion = TargetedMisclassification(self.y_target)
            _, advx, _ = self.foolbox_attack(
                model=foolbox_model,
                inputs=samples,
                criterion=criterion,
                epsilons=self.epsilon,
            )
            adversarials.append(advx)
            original_labels.append(labels)
        adversarials = torch.vstack(adversarials)
        original_labels = torch.hstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        adversarial_loader = DataLoader(
            adversarial_dataset, batch_size=data_loader.batch_size
        )
        return adversarial_loader
