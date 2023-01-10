import torch

from src.models.preprocessing.preprocessing import Preprocessing


class IdentityPreprocessing(Preprocessing):

	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
		return x

	def invert(self, x: torch.Tensor) -> torch.Tensor:
		return x
