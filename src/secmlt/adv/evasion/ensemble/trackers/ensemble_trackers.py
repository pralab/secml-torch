from secmlt.trackers import Tracker, PredictionTracker
import torch
from secmlt.trackers.trackers import MULTI_SCALAR
from typing import Union


class EnsemblePredictionTracker(Tracker):
    """Tracker for ensemble model predictions."""

    def __init__(self) -> None:
        """Create prediction tracker."""
        super().__init__("EnsemblePrediction", MULTI_SCALAR)
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
        """
        Track the sample-wise model predictions at the current iteration.

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
        predictions = scores.data.argmax(dim=-1).T
        self.tracked.append(predictions.data)


class MajorityVotingPredictionTracker(PredictionTracker):
    """Tracker for ensemble model predictions."""

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
        Track the sample-wise model predictions at the current iteration.

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
        predictions = scores.data.argmax(dim=-1)
        if predictions.ndim > 0:  # RawEnsemblingFunction -> apply majority voting
            predictions = predictions.mode(dim=0).values
        self.tracked.append(predictions)


class EnsembleScoresTracker(Tracker):
    """Tracker for model scores."""

    def __init__(self, y: Union[int, torch.Tensor] = 1) -> None:
        """Create scores tracker."""
        super().__init__("Scores", MULTI_SCALAR)
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
        """
        Track the sample-wise model scores at the current iteration.

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
        if scores.ndim > 0:
            self.tracked.append(scores.transpose(0, 1).data[..., self.y])
        else:
            self.tracked.append(scores.data[..., self.y])

class AvgEnsembleScoresTracker(Tracker):
    """Tracker for model scores."""

    def __init__(self, y: Union[int, torch.Tensor] = None) -> None:
        """Create scores tracker."""
        super().__init__("AvgScores", MULTI_SCALAR)
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
        """
        Track the sample-wise model scores at the current iteration.

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
        if scores.ndim > 0:  # RawEnsemblingFunction -> average scores
            scores = scores.mean(dim=0)
            self.tracked.append(scores.data if self.y is None
                                else scores.data[..., self.y])
        else:
            self.tracked.append(scores.data if self.y is None
                                else scores.data[..., self.y])
