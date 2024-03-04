import torch

from secmlt.adv.evasion.perturbation_models import PerturbationModels


class Initializer:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        init = torch.zeros_like(x)
        return init


class RandomLpInitializer(Initializer):
    def __init__(self, radius: torch.Tensor, perturbation_model: PerturbationModels):
        self.radius = radius
        self.perturbation_model = perturbation_model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented")
