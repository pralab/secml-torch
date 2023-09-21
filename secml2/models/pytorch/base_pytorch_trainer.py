import torch.nn
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torch.utils.data import DataLoader

from secml2.models.base_trainer import BaseTrainer


class BasePyTorchTrainer(BaseTrainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        loss: torch.nn.Module = None,
        scheduler: _LRScheduler = None,
    ):
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
        self._scheduler = scheduler

    def train(self, model: torch.nn.Module, dataloader: DataLoader):
        device = next(model.parameters()).device
        model = model.train()
        for epoch in range(self._epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                self._optimizer.zero_grad()
                outputs = model(x)
                loss = self._loss(outputs, y)
                loss.backward()
                self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
        return model
