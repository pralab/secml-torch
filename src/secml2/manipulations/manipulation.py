from abc import ABC
from typing import Tuple

import torch

from secml2.optimization.constraints import Constraint


class Manipulation(ABC):
    def __init__(
        self,
        domain_constraints: list[Constraint],
        perturbation_constraints: list[Constraint],
    ):
        self.domain_constraints = domain_constraints
        self.perturbation_constraints = perturbation_constraints

    def _apply_domain_constraints(self, x: torch.Tensor) -> torch.Tensor:
        for constraint in self.domain_constraints:
            x = constraint(x)
        return x

    def _apply_perturbation_constraints(self, delta: torch.Tensor) -> torch.Tensor:
        for constraint in self.perturbation_constraints:
            delta = constraint(delta)
        return delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta.data = self._apply_perturbation_constraints(delta.data)
        x_adv, delta = self._apply_manipulation(x, delta)
        x_adv.data = self._apply_domain_constraints(x_adv.data)
        return x_adv, delta


class AdditiveManipulation(Manipulation):
    def _apply_manipulation(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + delta, delta
