from secmlt.adv.evasion.perturbation_models import PerturbationModels
from secmlt.trackers.trackers import IMAGE, Tracker
import torch


class SampleTracker(Tracker):
    def __init__(self) -> None:
        super().__init__("Sample", IMAGE)

        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(x_adv)


class GradientsTracker(Tracker):
    def __init__(self) -> None:
        super().__init__("Grad", IMAGE)

        self.tracked = []

    def track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        self.tracked.append(grad)
