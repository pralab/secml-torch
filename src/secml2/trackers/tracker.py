from abc import ABC

import torch
from torch.utils.tensorboard import SummaryWriter

from secml2.adv.evasion.perturbation_models import PerturbationModels


class Tracker(ABC):
    def __init__(self, name):
        self.name = name

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
    ):
        ...

    def get(self):
        ...

    def get_last_tracked(self):
        ...


class LossTracker(Tracker):
    def __init__(self):
        super().__init__("Loss")
        self.loss = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
    ):
        self.loss.append(loss.data.item())

    def get(self):
        return self.loss

    def get_last_tracked(self):
        if self.loss:
            return self.loss[-1]
        return None


class ScoresTracker(Tracker):
    def __init__(self):
        super().__init__("Scores")
        self.scores = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
    ):
        self.scores.append(scores.data)

    def get(self):
        return self.scores

    def get_last_tracked(self):
        if self.scores:
            return self.scores[-1]
        return None


class PredictionTracker(Tracker):
    def __init__(self):
        super().__init__("Prediction")
        self.predictions = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
    ):
        self.predictions.append(scores.data.argmax(dim=1))

    def get(self):
        return self.predictions

    def get_last_tracked(self):
        if self.predictions:
            return self.predictions[-1]
        return None


class GradientTracker(Tracker):
    def __init__(self, p: PerturbationModels = PerturbationModels.L2):
        super().__init__("Grad Norm")

        perturbations_models = {
            PerturbationModels.L0: 0,
            PerturbationModels.L1: 1,
            PerturbationModels.L2: 2,
            PerturbationModels.LINF: float("inf"),
        }
        self.p = perturbations_models[p]
        self.grad_norms = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
    ):
        grad: torch.Tensor = delta.grad
        if grad is None:
            self.grad_norms.append(None)
        norm = grad.data.flatten(start_dim=1).norm(p=self.p, dim=1)
        self.grad_norms.append(norm)

    def get(self):
        return self.grad_norms

    def get_last_tracked(self):
        if self.grad_norms:
            return self.grad_norms[-1]
        return None


class TensorboardTracker(Tracker):
    def __init__(self, logdir: str, trackers: list[Tracker] = None):
        super().__init__("Tensorboard")
        if trackers is None:
            trackers = [
                LossTracker(),
                PredictionTracker(),
                GradientTracker(),
            ]
        self.writer = SummaryWriter(log_dir=logdir)
        self.trackers = trackers

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        delta: torch.Tensor,
    ):
        for tracker in self.trackers:
            tracker.track(iteration, loss, scores, delta)
            tracked_value = tracker.get_last_tracked()
            if isinstance(tracked_value, torch.Tensor):
                for i, sample in enumerate(tracked_value):
                    self.writer.add_scalar(
                        f"{tracker.name} #{i}", sample, global_step=iteration
                    )
            else:
                self.writer.add_scalar(
                    f"{tracker.name}", tracked_value, global_step=iteration
                )

    def get_last_tracked(self):
        return {tracker.name: tracker.get() for tracker in self.trackers}
