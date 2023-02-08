from typing import Optional

import torch

from src.adv.evasion.perturbation_models import PerturbationModels
from src.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from src.adv.backends import Backends
from src.adv.evasion.base_evasion_attack import (
	BaseEvasionAttack,
	BaseEvasionAttackCreator,
)
from src.models.base_model import BaseModel

from foolbox.attacks.projected_gradient_descent import (
	L1ProjectedGradientDescentAttack,
	L2ProjectedGradientDescentAttack,
	LinfProjectedGradientDescentAttack,
)
from foolbox.models.pytorch import PyTorchModel
from foolbox.criteria import Misclassification, TargetedMisclassification


class PGD(BaseEvasionAttackCreator):
	def __new__(
			cls,
			perturbation_model: str,
			epsilon: float,
			num_steps: int,
			step_size: float,
			random_start: bool,
			y_target: Optional[int] = None,
			lb: float = 0.0,
			ub: float = 1.0,
			backend: str = Backends.FOOLBOX,
			**kwargs
	):
		cls.check_perturbation_model_available(perturbation_model)
		implementation = cls.get_implementation(backend)
		return implementation(
			perturbation_model, epsilon, num_steps, step_size, random_start, y_target, lb, ub, **kwargs
		)

	@staticmethod
	def get_foolbox_implementation():
		return PGDFoolbox


class PGDFoolbox(BaseEvasionAttack):
	def __init__(
			self,
			perturbation_model: str,
			epsilon: float,
			num_steps: int,
			step_size: float,
			random_start: bool,
			y_target: Optional[int] = None,
			lb: float = 0.0,
			ub: float = 1.0,
			**kwargs
	) -> None:
		perturbation_models = {
			PerturbationModels.L1: L1ProjectedGradientDescentAttack,
			PerturbationModels.L2: L2ProjectedGradientDescentAttack,
			PerturbationModels.LINF: LinfProjectedGradientDescentAttack,
		}
		foolbox_attack_cls = perturbation_models.get(perturbation_model, None)
		if foolbox_attack_cls is None:
			raise NotImplementedError(
				"This threat model is not implemented in foolbox."
			)

		self.epsilon = epsilon
		self.lb = lb
		self.ub = ub
		self.y_target = y_target

		self.foolbox_attack = foolbox_attack_cls(
			abs_stepsize=step_size, steps=num_steps, random_start=random_start
		)
		super().__init__()

	def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
		# TODO get here the correct model
		if not isinstance(model, BasePytorchClassifier):
			raise NotImplementedError("Model type not supported.")
		device = model.get_device()
		foolbox_model = PyTorchModel(
			model.model, (self.lb, self.ub), device=device
		)
		adversarials = []
		original_labels = []
		for samples, labels in data_loader:
			samples, labels = samples.to(device), labels.to(device)
			if self.y_target is None:
				criterion = Misclassification(labels)
			else:
				criterion = TargetedMisclassification(self.y_target)
			_, advx, _ = self.foolbox_attack(model=foolbox_model, inputs=samples, criterion=criterion,
			                                 epsilons=self.epsilon)
			adversarials.append(advx)
			original_labels.append(labels)
		adversarials = torch.vstack(adversarials)
		original_labels = torch.hstack(original_labels)
		adversarial_dataset = TensorDataset(adversarials, original_labels)
		adversarial_loader = DataLoader(adversarial_dataset, batch_size=data_loader.batch_size)
		return adversarial_loader
