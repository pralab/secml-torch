import torch


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    acc = (y_pred.type(y_true.dtype) == y_true).mean()
    return acc


class Accuracy(object):
    def __init__(self):
        self._num_samples = 0
        self._accumulated_accuracy = 0.0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self._num_samples += y_true.shape[0]
        self._accumulated_accuracy += y_pred.type(y_true.dtype) == y_true

    def compute(self):
        return self._accumulated_accuracy / self._num_samples
