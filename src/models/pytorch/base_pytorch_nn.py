import torch
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel
from src.models.preprocessing.preprocessing import Preprocessing
from src.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer


class BasePytorchClassifier(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: Preprocessing = None,
        trainer: BasePyTorchTrainer = None,
    ):
        super().__init__(preprocessing=preprocessing)
        self._model: torch.nn.Module = model
        self._trainer = trainer

    @property
    def model(self):
        return self._model

    def get_device(self):
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            # probably sklearn wrapped module
            return "cpu"

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.decision_function(x)
        labels = torch.argmax(scores, dim=-1)
        return labels

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the decision function of the model.
        Parameters
        ----------
        x : input samples

        Returns
        -------
        output of decision function
        """
        x = x.to(device=self.get_device())
        return self._model(x)

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        The functions computes gradients of class with label y w.r.t. x.
        Gradients are computed in batch.
        Parameters
        ----------
        x : input samples
        y : class label

        Returns
        -------
        Gradient of class y w.r.t. input x
        """
        x = x.clone().requires_grad_()
        if x.grad is not None:
            x.grad.zero_()
        output = self.decision_function(x)
        output = output[:, y].sum()
        output.backward()
        grad = x.grad
        return grad

    def train(self, dataloader: DataLoader):
        if self._trainer is None:
            raise ValueError("Cannot train without a trainer.")
        return self._trainer.train(self._model, dataloader)
