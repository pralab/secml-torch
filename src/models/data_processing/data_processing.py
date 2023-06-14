from abc import ABC, abstractmethod

import torch


class DataProcessing(ABC):
    @abstractmethod
    def process(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)
