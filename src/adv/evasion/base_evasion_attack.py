from abc import abstractmethod

from torch.utils.data import DataLoader

from src.models.base_model import BaseModel


class BaseEvasionAttack:
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
