"""PyTorch model trainers with early stopping."""

import torch.nn
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class EarlyStoppingPyTorchTrainer(BasePyTorchTrainer):
    """Trainer for PyTorch models."""

    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int = 5,
                 loss: torch.nn.Module = None, scheduler: _LRScheduler = None) -> None:
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
        super().__init__(optimizer, epochs, loss, scheduler)
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
        self._scheduler = scheduler

    def fit(self, model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            patience: int) -> torch.nn.Module:
        """
        Train model with given loaders and early stopping.

        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be trained.
        train_loader : DataLoader
            Train data loader.
        val_loader : DataLoader
            Validation data loader.
        patience : int
            Number of epochs to wait before early stopping.

        Returns
        -------
        torch.nn.Module
            Trained model.
        """
        best_loss = float("inf")
        best_model = None
        patience_counter = 0
        for _ in range(self._epochs):
            model = self.train(model, train_loader)
            val_loss = self.validate(model, val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        return best_model

    def train(self,
              model: torch.nn.Module,
              dataloader: DataLoader) -> torch.nn.Module:
        """
        Train model for one epoch with given loader.

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

    def validate(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader) -> torch.nn.Module:
        """
        Validate model with given loader.

        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be balidated.
        dataloader : DataLoader
            Validation data loader.

        Returns
        -------
        float
            Validation loss of the model.
        """
        running_loss = 0
        device = next(model.parameters()).device
        model = model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = self._loss(outputs, y)
                running_loss += loss.item()
        return loss
