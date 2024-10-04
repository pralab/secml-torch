"""Image-specific trackers."""

import torch
from secmlt.trackers.trackers import IMAGE, Tracker


class SampleTracker(Tracker):
    """Tracker for adversarial images."""

    def __init__(self) -> None:
        """Create adversarial image tracker."""
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
        """
        Track the adversarial examples at the current iteration as images.

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
        self.tracked.append(x_adv)


class GradientsTracker(Tracker):
    """Tracker for gradient images."""

    def __init__(self) -> None:
        """Create gradients tracker."""
        super().__init__(name="Grad", tracker_type=IMAGE)

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
        """
        Track the gradients at the current iteration as images.

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
        self.tracked.append(grad)
