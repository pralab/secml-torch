import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class SklearnDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y must have the same number of rows (mismatch {x.shape} and {y.shape})')
        self._x: np.ndarray = x
        self._y: np.ndarray = y

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @x.setter
    def x(self, data: np.ndarray):
        self._x = data

    @y.setter
    def y(self, data: np.ndarray):
        self._y = data

    def __getitem__(self, index) -> T_co:
        return self._x[index, ...], self._y[index, ...]

    def __len__(self) -> int:
        return self._x.shape[0]
