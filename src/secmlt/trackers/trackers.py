"""Trackers for attack metrics."""

from abc import ABC, abstractmethod
from collections.abc import Callable
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
        self._batches = []
        self.requires_grad = False

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

    def init_tracking(self) -> None:
        """Initialize tracking for a new batch (clears the per-batch buffer)."""
        if hasattr(self, "tracked") and isinstance(self.tracked, list):
            self.tracked = []
        elif hasattr(self, "tracked"):
            self.tracked = None

    def end_tracking(self) -> None:
        """Finalize the current batch and append its history to `_batches`."""
        if (
            hasattr(self, "tracked")
            and isinstance(self.tracked, list)
            and len(self.tracked) > 0
        ):
            if not hasattr(self, "_batches"):
                self._batches = []
            self._batches.append(torch.stack(self.tracked, -1))
            self.tracked = []

    def reset(self) -> None:
        """Clear all tracking history across all batches."""
        if hasattr(self, "tracked") and isinstance(self.tracked, list):
            self.tracked = []
        elif hasattr(self, "tracked"):
            self.tracked = None
        self._batches = []

    def get(self) -> torch.Tensor:
        """
        Get the current tracking history.

        Returns
        -------
        torch.Tensor
            History of tracked parameters. When multiple batches were tracked,
            returns a tensor where batches are concatenated along the sample
            dimension (dim=0) and iterations are along the last dimension.
        """
        if not self._batches:
            if (
                hasattr(self, "tracked")
                and isinstance(self.tracked, list)
                and len(self.tracked) > 0
            ):
                return torch.stack(self.tracked, -1)
            return torch.empty(0)
        if len(self._batches) == 1:
            return self._batches[0]
        return torch.cat(self._batches, dim=0)

    def get_last_tracked(self) -> Union[None, torch.Tensor]:
        """
        Get last element tracked.

        Returns
        -------
        None | torch.Tensor
            Returns the last tracked element if anything was tracked.
        """
        # Prefer the most recent value from the ongoing batch
        if (
            hasattr(self, "tracked")
            and isinstance(self.tracked, list)
            and len(self.tracked) > 0
        ):
            return self.tracked[-1]
        # Otherwise take the last iteration from the last finalized batch
        if hasattr(self, "_batches") and len(self._batches) > 0:
            return self._batches[-1][..., -1]
        return None


class LossTracker(Tracker):
    """Tracker for attack loss."""

    def __init__(self, loss_fn: Callable | None = None) -> None:
        """Create loss tracker.

        Parameters
        ----------
        loss_fn : callable | None, optional
            Per-sample loss function accepting ``(scores, labels)``.
            When this tracker is used with ``ModelTracker`` and no loss
            is provided by the attack loop, this function is used to
            compute losses from model outputs. By default a per-sample
            cross-entropy is used.
        """
        super().__init__("Loss")
        self.tracked = []
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else torch.nn.CrossEntropyLoss(reduction="none")
        )

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
        loss : torch.Tensor | None
            The value of the (per-sample) loss of the attack.
            The model can optionally pass None for the loss,
            in which case this tracker will attempt to compute
            the loss using the provided loss_fn.
            If loss_fn is not provided, it will skip tracking for that iteration.
        scores : torch.Tensor
            The output scores from the model.
        x_adv : torch.tensor
            The adversarial examples at the current iteration.
        delta : torch.Tensor
            The adversarial perturbations at the current iteration.
        grad : torch.Tensor
            The gradient of delta at the given iteration.
        """
        if loss is None:
            return
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


class SampleTracker(Tracker):
    """Generic tracker for adversarial samples."""

    def __init__(self, tracker_type: str = MULTI_SCALAR) -> None:
        """
        Create sample tracker.

        Parameters
        ----------
        tracker_type : str, optional
            Tracked value type used by integrations (e.g. tensorboard),
            by default MULTI_SCALAR.
        """
        super().__init__("Sample", tracker_type)
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
        Track adversarial examples at the current iteration.

        Parameters
        ----------
        iteration : int
            The attack iteration number.
        loss : torch.Tensor
            The value of the (per-sample) loss of the attack.
        scores : torch.Tensor
            The output scores from the model.
        x_adv : torch.Tensor
            The adversarial examples at the current iteration.
        delta : torch.Tensor
            The adversarial perturbations at the current iteration.
        grad : torch.Tensor
            The gradient of delta at the given iteration.
        """
        if self.tracked_type == SCALAR and x_adv.ndim > 1:
            msg = (
                "SampleTracker with tracker_type='scalar' expects per-sample "
                "0D tensors. Received non-scalar sample values. Use "
                "tracker_type='multiple_scalars' for vectors or "
                "ImageSampleTracker/tracker_type='image' for images."
            )
            raise ValueError(msg)
        self.tracked.append(x_adv)


class GradientsTracker(Tracker):
    """Generic tracker for gradients."""

    def __init__(self, tracker_type: str = MULTI_SCALAR) -> None:
        """
        Create gradients tracker.

        Parameters
        ----------
        tracker_type : str, optional
            Tracked value type used by integrations (e.g. tensorboard),
            by default MULTI_SCALAR.
        """
        super().__init__(name="Grad", tracker_type=tracker_type)
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
        Track the gradients at the current iteration.

        Parameters
        ----------
        iteration : int
            The attack iteration number.
        loss : torch.Tensor
            The value of the (per-sample) loss of the attack.
        scores : torch.Tensor
            The output scores from the model.
        x_adv : torch.Tensor
            The adversarial examples at the current iteration.
        delta : torch.Tensor
            The adversarial perturbations at the current iteration.
        grad : torch.Tensor | None
            The gradient of delta at the given iteration.
            The model can optionally pass None for the gradient,
            in which case this tracker will simply skip tracking for that iteration.
        """
        if self.tracked_type == SCALAR and grad.ndim > 1:
            msg = (
                "GradientsTracker with tracker_type='scalar' expects per-sample "
                "0D tensors. Received non-scalar sample values. Use "
                "tracker_type='multiple_scalars' for vectors or "
                "ImageGradientsTracker/tracker_type='image' for images."
            )
            raise ValueError(msg)
        self.tracked.append(grad)


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
        self.requires_grad = True

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
        if grad is None:
            return
        norm = grad.data.flatten(start_dim=1).norm(p=self.p, dim=1)
        self.tracked.append(norm)
