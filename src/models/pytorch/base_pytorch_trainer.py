import torch.nn
from torch.optim.lr_scheduler import _LRScheduler  # noqa

from src.metrics.classification import Accuracy
from src.models.base_trainer import BaseTrainer


class BasePyTorchTrainer(BaseTrainer):
	def __init__(self, optimizer: torch.optim.Optimizer, epochs: int = 5, loss: torch.nn.Module = None,
				 scheduler: _LRScheduler = None, device: str = None):
		self._epochs = epochs
		self._optimizer = optimizer
		self._loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
		self._scheduler = scheduler
		device_available = "cuda" if torch.cuda.is_available() else "cpu"
		self._device = device if device is not None else device_available

	def train(self, model, dataloader):
		model = model.to(self._device).train()
		for epoch in range(self._epochs):
			for batch_idx, (x, y) in enumerate(dataloader):
				x, y = x.to(self._device), y.to(self._device)
				self._optimizer.zero_grad()
				outputs = model(x)
				loss = self._loss(outputs, y)
				loss.backward()
				self._optimizer.step()
			if self._scheduler is not None:
				self._scheduler.step()
		return model

	def test(self, model, dataloader):
		model = model.to(self._device).eval()
		acc = Accuracy()
		for batch_idx, (x, y) in enumerate(dataloader):
			x, y = x.to(self._device), y.to(self._device)
			outputs = model(x)
			y_pred = outputs.max(dim=1)
			acc(y_pred, y)
		accuracy = acc.compute()
		return accuracy
