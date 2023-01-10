from src.models.base_trainer import BaseTrainer


class BaseTensorflowTrainer(BaseTrainer):
	def __init__(self):
		pass

	def train(self, model, dataloader):
		pass

	def test(self, model, dataloader):
		pass