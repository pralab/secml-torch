from abc import ABCMeta, abstractmethod


class BaseTrainer(metaclass=ABCMeta):
	@abstractmethod
	def train(self, model, dataloader):
		pass

	@abstractmethod
	def test(self, model, dataloader):
		pass