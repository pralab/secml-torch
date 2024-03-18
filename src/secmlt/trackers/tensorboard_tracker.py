"""Tensorboard tracking utilities."""

import torch
from secmlt.trackers.trackers import (
    IMAGE,
    MULTI_SCALAR,
    SCALAR,
    GradientNormTracker,
    LossTracker,
    Tracker,
)
from torch.utils.tensorboard import SummaryWriter


class TensorboardTracker(Tracker):
    """Tracker for Tensorboard. Uses other trackers as subscribers."""

    def __init__(self, logdir: str, trackers: list[Tracker] | None = None) -> None:
        """
        Create tensorboard tracker.

        Parameters
        ----------
        logdir : str
            Folder to store tensorboard logs.
        trackers : list[Tracker] | None, optional
            List of trackers subsctibed to the updates, by default None.
        """
        super().__init__(name="Tensorboard")
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
    ) -> None:
        """
        Update all subscribed trackers.

        Parameters
        ----------
        iteration : int
            The attack iteration number.
        loss : torch.Tensor
            The value of the (per-sample) loss of the attack.
        scores : torch.Tensor
            The output scores from the model.
        x_adv : torch.tensor
            The adversarial examples at the current iteration.
        delta : torch.Tensor
            The adversarial perturbations at the current iteration.
        grad : torch.Tensor
            The gradient of delta at the given iteration.
        """
        for tracker in self.trackers:
            tracker.track(iteration, loss, scores, x_adv, delta, grad)
            tracked_value = tracker.get_last_tracked()
            for i, sample in enumerate(tracked_value):
                if tracker.tracked_type == SCALAR:
                    self.writer.add_scalar(
                        f"Sample #{i}/{tracker.name}",
                        sample,
                        global_step=iteration,
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
                        f"Sample #{i}/{tracker.name}",
                        sample,
                        global_step=iteration,
                    )

    def get_last_tracked(self) -> NotImplementedError:
        """Not implemented for this tracker."""
        return NotImplementedError(
            "Last tracked value is not available for this tracker.",
        )
