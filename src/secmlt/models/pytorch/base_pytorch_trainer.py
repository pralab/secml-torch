"""PyTorch model trainers."""

import torch.nn
from secmlt.models.base_trainer import BaseTrainer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class BasePyTorchTrainer(BaseTrainer):
    """Trainer for PyTorch models."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        loss: torch.nn.Module = None,
        scheduler: _LRScheduler = None,
    ) -> None:
        """
        Create PyTorch trainer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to use for training the model.
        epochs : int, optional
            Number of epochs, by default 5.
        loss : torch.nn.Module, optional
            Loss to minimize, by default None.
        scheduler : _LRScheduler, optional
            Scheduler for the optimizer, by default None.
        """
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
        self._scheduler = scheduler

    def train(self, model: torch.nn.Module, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train model with given loader.

        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be trained.
        dataloader : DataLoader
            Train data loader.

        Returns
        -------
        torch.nn.Module
            Trained model.
        """
        device = next(model.parameters()).device
        model = model.train()
        for _ in range(self._epochs):
            for _, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                self._optimizer.zero_grad()
                outputs = model(x)
                loss = self._loss(outputs, y)
                loss.backward()
                self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
        return model
