from typing import List, Type
from secmlt.trackers.trackers import (
    IMAGE,
    MULTI_SCALAR,
    SCALAR,
    GradientNormTracker,
    LossTracker,
    Tracker,
)
import torch
from torch.utils.tensorboard import SummaryWriter


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
        x_adv: torch.tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
    ):
        for tracker in self.trackers:
            tracker.track(iteration, loss, scores, x_adv, delta, grad)
            tracked_value = tracker.get_last_tracked()
            for i, sample in enumerate(tracked_value):
                if tracker.tracked_type == SCALAR:
                    self.writer.add_scalar(
                        f"Sample #{i}/{tracker.name}", sample, global_step=iteration
                    )
                elif tracker.tracked_type == MULTI_SCALAR:
                    self.writer.add_scalars(
                        main_tag=f"Sample #{i}/{tracker.name}",
                        tag_scalar_dict={
                            f"Sample #{i}/{tracker.name}{j}": v
                            for j, v in enumerate(sample)
                        },
                        global_step=iteration,
                    )
                elif tracker.tracked_type == IMAGE:
                    self.writer.add_image(
                        f"Sample #{i}/{tracker.name}", sample, global_step=iteration
                    )

    def get_last_tracked(self):
        return NotImplementedError(
            "Last tracked value is not available for this tracker."
        )
