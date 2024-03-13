from secmlt.optimization.random_perturb import RandomPerturb
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
        self.initializer = RandomPerturb(p=self.perturbation_model, epsilon=self.radius)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.initializer(x)
