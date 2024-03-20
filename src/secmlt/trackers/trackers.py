"""Trackers for attack metrics."""

from abc import ABC, abstractmethod
from typing import Union

import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

SCALAR = "scalar"
IMAGE = "image"
MULTI_SCALAR = "multiple_scalars"


class Tracker(ABC):
    """Class implementing the trackers for the attacks."""

    def __init__(self, name: str, tracker_type: str = SCALAR) -> None:
        """
        Create tracker.

        Parameters
        ----------
        name : str
            Tracker name.
        tracker_type : str, optional
            Type of tracker (mostly used for tensorboard functionalities),
            by default SCALAR. Available: SCALAR, IMAGE, MULTI_SCALAR.
        """
        self.name = name
        self.tracked = None
        self.tracked_type = tracker_type

    @abstractmethod
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
        Track the history of given attack observable parameters.

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

    def get(self) -> torch.Tensor:
        """
        Get the current tracking history.

        Returns
        -------
        torch.Tensor
            History of tracked parameters.
        """
        return torch.stack(self.tracked, -1)

    def get_last_tracked(self) -> Union[None, torch.Tensor]:
        """
        Get last element tracked.

        Returns
        -------
        None | torch.Tensor
            Returns the last tracked element if anything was tracked.
        """
        if self.tracked is not None:
            return self.get()[..., -1]  # return last tracked value
        return None


class LossTracker(Tracker):
    """Tracker for attack loss."""

    def __init__(self) -> None:
        """Create loss tracker."""
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
        """
        Track the sample-wise loss of the attack at the current iteration.

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
        self.tracked.append(loss.data)


class ScoresTracker(Tracker):
    """Tracker for model scores."""

    def __init__(self, y: Union[int, torch.Tensor] = None) -> None:
        """Create scores tracker."""
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
        if self.y is None:
            self.tracked.append(scores.data)
        else:
            self.tracked.append(scores.data[..., self.y])


class PredictionTracker(Tracker):
    """Tracker for model predictions."""

    def __init__(self) -> None:
        """Create prediction tracker."""
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
        self.tracked.append(scores.data.argmax(dim=1))


class PerturbationNormTracker(Tracker):
    """Tracker for perturbation norm."""

    def __init__(self, p: LpPerturbationModels = LpPerturbationModels.L2) -> None:
        """
        Create perturbation norm tracker.

        Parameters
        ----------
        p : LpPerturbationModels, optional
            Perturbation model to compute the norm, by default LpPerturbationModels.L2.
        """
        super().__init__("PertNorm")
        self.p = LpPerturbationModels.get_p(p)
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
        Track the perturbation norm at the current iteration.

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
        self.tracked.append(delta.flatten(start_dim=1).norm(p=self.p, dim=-1))


class GradientNormTracker(Tracker):
    """Tracker for gradients."""

    def __init__(self, p: LpPerturbationModels = LpPerturbationModels.L2) -> None:
        """
        Create gradient norm tracker.

        Parameters
        ----------
        p : LpPerturbationModels, optional
            Perturbation model to compute the norm, by default LpPerturbationModels.L2.
        """
        super().__init__("GradNorm")

        self.p = LpPerturbationModels.get_p(p)
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
        Track the sample-wise gradient of the loss w.r.t delta.

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
        norm = grad.data.flatten(start_dim=1).norm(p=self.p, dim=1)
        self.tracked.append(norm)
