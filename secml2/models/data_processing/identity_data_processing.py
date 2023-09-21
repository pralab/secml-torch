import torch

from secml2.models.data_processing.data_processing import DataProcessing


class IdentityDataProcessing(DataProcessing):
    def process(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x
