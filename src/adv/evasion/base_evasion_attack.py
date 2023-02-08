from abc import abstractmethod, ABC
from typing import Callable

from torch.utils.data import DataLoader

from src.adv.backends import Backends
from src.adv.evasion.threat_models import ThreatModels
from src.models.base_model import BaseModel


class BaseEvasionAttackCreator(ABC):

	def get_implementation(self, backend: str) -> Callable:
		implementations = {
			Backends.FOOLBOX: self.get_foolbox_implementation,
			Backends.NATIVE: self.get_native_implementation,
		}
		if backend not in implementations:
			raise NotImplementedError('Unsupported or not-implemented backend.')
		return implementations[backend]

	@classmethod
	def check_threat_model_available(cls, threat_model: str):
		if not ThreatModels.is_threat_model_available(threat_model):
			raise NotImplementedError('Unsupported or not-implemented threat model.')

	def get_foolbox_implementation(self):
		raise NotImplementedError('Foolbox implementation not available.')

	def get_native_implementation(self):
		raise NotImplementedError('Native implementation not available.')

	@abstractmethod
	def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
		"""
		Compute the attack against the model, using the input data.
		It returns a dataset with the adversarial examples and the original labels
		:param model: model to test
		:type model: BaseModel
		:param data_loader: input data
		:type data_loader: DataLoader
		:return: Data loader with adversarial examples and original labels
		:rtype: DataLoader
		"""
		...
