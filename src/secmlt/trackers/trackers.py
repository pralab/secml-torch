from abc import ABC
from typing import Union
from secmlt.adv.evasion.perturbation_models import PerturbationModels

import torch

SCALAR = "scalar"
IMAGE = "image"
MULTI_SCALAR = "multiple_scalars"


class Tracker(ABC):
    def __init__(self, name, tracker_type=SCALAR) -> None:
        self.name = name
        self.tracked = None
        self.tracked_type = tracker_type

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None: ...

    def get(self) -> torch.Tensor:
        return torch.stack(self.tracked, -1)

    def get_last_tracked(self) -> Union[None, torch.Tensor]:
        if self.tracked is not None:
            return self.get()[..., -1]  # return last tracked value
        return None


class LossTracker(Tracker):
    def __init__(self) -> None:
        super().__init__("Loss")
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(loss.data)


class ScoresTracker(Tracker):
    def __init__(self, y: Union[int, torch.Tensor] = None) -> None:
        if y is None:
            super().__init__("Scores", MULTI_SCALAR)
        else:
            super().__init__("Scores")
        self.y = y
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        if self.y is None:
            self.tracked.append(scores.data)
        else:
            self.tracked.append(scores.data[..., self.y])


class PredictionTracker(Tracker):
    def __init__(self) -> None:
        super().__init__("Prediction")
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(scores.data.argmax(dim=1))


class PerturbationNormTracker(Tracker):
    def __init__(self, p: PerturbationModels = PerturbationModels.L2) -> None:
        super().__init__("PertNorm")
        self.p = PerturbationModels.get_p(p)
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(delta.flatten(start_dim=1).norm(p=self.p, dim=-1))


class GradientNormTracker(Tracker):
    def __init__(self, p: PerturbationModels = PerturbationModels.L2) -> None:
        super().__init__("GradNorm")

        self.p = PerturbationModels.get_p(p)
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        norm = grad.data.flatten(start_dim=1).norm(p=self.p, dim=1)
        self.tracked.append(norm)
