from abc import ABC
from typing import Union, List, Type
from secml2.adv.evasion.perturbation_models import PerturbationModels

import torch
from torch.utils.tensorboard import SummaryWriter


class Tracker(ABC):
    def __init__(self, name) -> None:
        self.name = name
        self.tracked = None

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
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
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(loss.data)


class ScoresTracker(Tracker):
    def __init__(self) -> None:
        super().__init__("Scores")
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(scores.data)


class PredictionTracker(Tracker):
    def __init__(self) -> None:
        super().__init__("Prediction")
        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
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
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        norm = grad.data.flatten(start_dim=1).norm(p=self.p, dim=1)
        self.tracked.append(norm)


class TensorboardTracker(Tracker):
    def __init__(self, logdir: str, trackers: List[Type[Tracker]] = None):
        super().__init__("Tensorboard")
        if trackers is None:
            trackers = [
                LossTracker(),
                GradientNormTracker(),
            ]
        self.writer = SummaryWriter(log_dir=logdir)
        self.trackers = trackers

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ):
        for tracker in self.trackers:
            tracker.track(iteration, loss, scores, delta, grad)
            tracked_value = tracker.get_last_tracked()
            for i, sample in enumerate(tracked_value):
                self.writer.add_scalar(
                    f"Sample #{i}/{tracker.name}", sample, global_step=iteration
                )

    def get_last_tracked(self):
        return NotImplementedError(
            "Last tracked value is not available for this tracker."
        )
