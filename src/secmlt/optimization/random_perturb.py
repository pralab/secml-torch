from abc import ABC, abstractmethod
from secmlt.adv.evasion.perturbation_models import PerturbationModels
from secmlt.optimization.constraints import (
    L0Constraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
import torch
from torch.distributions.laplace import Laplace


class RandomPerturbBase(ABC):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, x):
        perturbations = self.get_perturb(x)
        perturbations = self.constraint(
            radius=self.epsilon, center=torch.zeros_like(perturbations)
        ).project(perturbations)
        return perturbations

    @abstractmethod
    def get_perturb(self, x): ...

    @abstractmethod
    def constraint(self, x): ...


class RandomPerturbLinf(RandomPerturbBase):
    def get_perturb(self, x):
        x = torch.randn_like(x)
        return x

    @property
    def constraint(self):
        return LInfConstraint


class RandomPerturbL1(RandomPerturbBase):
    def __init__(self, epsilon):
        super().__init__(epsilon)

    def get_perturb(self, x):
        s = Laplace(0, 1)
        return s.sample(x.shape)

    @property
    def constraint(self):
        return L1Constraint


class RandomPerturbL2(RandomPerturbBase):
    def get_perturb(self, x):
        perturbations = torch.randn_like(x)
        return perturbations

    @property
    def constraint(self):
        return L2Constraint


class RandomPerturbL0(RandomPerturbBase):
    def get_perturb(self, x):
        perturbations = torch.randn_like(x)
        return perturbations.sign()

    @property
    def constraint(self):
        return L0Constraint


class RandomPerturb:
    def __new__(cls, p, epsilon) -> RandomPerturbBase:
        random_inits = {
            PerturbationModels.L0: RandomPerturbL0,
            PerturbationModels.L1: RandomPerturbL1,
            PerturbationModels.L2: RandomPerturbL2,
            PerturbationModels.LINF: RandomPerturbLinf,
        }
        selected = random_inits.get(p, None)
        if selected is not None:
            return selected(epsilon=epsilon)
        raise ValueError(
            "Random Perturbation not implemented for this perturbation model."
        )
