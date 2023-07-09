import math

import torch.linalg
from torch.nn.functional import normalize

from secml2.adv.evasion.perturbation_models import PerturbationModels


class GradientProcessing:
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        ...


class LinearProjectionGradientProcessing(GradientProcessing):
    def __init__(self, perturbation_model: str = PerturbationModels.L2):
        perturbations_models = {
            PerturbationModels.L1: 1,
            PerturbationModels.L2: 2,
            PerturbationModels.LINF: float("inf"),
        }
        if perturbation_model not in perturbations_models:
            raise ValueError(
                f"{perturbation_model} not included in normalizers. Available: {perturbations_models.values()}"
            )
        self.p = perturbations_models[perturbation_model]

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if self.p == 2:
            grad = normalize(grad.data, p=self.p, dim=0)
            return grad
        if self.p == float("inf"):
            return torch.sign(grad)
        raise NotImplementedError("Only L2 and LInf norms implemented now")
