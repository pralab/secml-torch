from torch.utils.data import DataLoader
import numpy as np


class SklearnDataLoader:
    """
    Converts a PyTorch DataLoader to a Sklearn Dataset with x and y.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._data_shape = next(iter(dataloader))[0].shape
        self._n_samples = len(dataloader.dataset)
        self._batch_size = dataloader.batch_size
        self.x = np.zeros(shape=(self._n_samples, *self._data_shape[1:]))
        self.y = np.zeros(shape=self._n_samples)
        self._get_samples()

    def _get_samples(self):
        for i, (samples, labels) in enumerate(self._dataloader):
            start_index = i * self._batch_size
            end_index = min((i + 1) * self._batch_size, self._n_samples)
            self.x[start_index:end_index] = samples.detach().cpu().numpy()
            self.y[start_index:end_index] = labels.detach().cpu().numpy()
        self.x = self.x.reshape(self._n_samples, -1)
        self.y = self.y.ravel()
