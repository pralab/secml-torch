import torch
from torch.utils.data import DataLoader

from secml2.models.base_model import BaseModel


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    acc = (y_pred.type(y_true.dtype) == y_true).mean()
    return acc


class Accuracy(object):
    def __init__(self):
        self._num_samples = 0
        self._accumulated_accuracy = 0.0

    def __call__(self, model: BaseModel, dataloader: DataLoader):
        for batch_idx, (x, y) in enumerate(dataloader):
            y_pred = model.predict(x).cpu().detach()
            self.accumulate(y_pred, y)
        accuracy = self.compute()
        return accuracy

    def accumulate(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self._num_samples += y_true.shape[0]
        self._accumulated_accuracy += torch.sum(
            y_pred.type(y_true.dtype).cpu() == y_true.cpu()
        )

    def compute(self):
        return self._accumulated_accuracy / self._num_samples
