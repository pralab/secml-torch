"""Model-level tracker for use with external attack libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class ModelTracker(BasePyTorchClassifier):
    """Passive tracker that wraps a model to intercept forward calls.

    This is an alternative to attack-level trackers, designed for use with
    external libraries (e.g., Foolbox, Adversarial Library) where modifying
    the attack loop is not possible. It registers a forward hook on the
    underlying ``nn.Module`` so that every forward pass feeds data to the
    subscribed trackers.

    Notes
    -----
    Gradient-aware trackers are supported without additional forward/backward
    queries by attaching a backward hook to the forward input tensor. The
    hook is executed when the caller performs its regular backward pass.
    """

    def __init__(
        self,
        model: BaseModel | torch.nn.Module,
        trackers: list[Tracker] | Tracker | None = None,
    ) -> None:
        """Create a model tracker."""
        self._hook_handle = None
        wrapped_model = self._ensure_wrapped(model)
        super().__init__(
            model=wrapped_model.model,
            preprocessing=wrapped_model._preprocessing,
            postprocessing=wrapped_model._postprocessing,
            trainer=getattr(wrapped_model, "_trainer", None),
        )
        if trackers is None:
            trackers = []
        elif not isinstance(trackers, list):
            trackers = [trackers]

        self._trackers: list[Tracker] = trackers
        self._iteration: int = 0
        self._x_orig: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._tracking: bool = False
        self._hook_handle = self._model.register_forward_hook(self._forward_hook)

    @staticmethod
    def _ensure_wrapped(model: BaseModel | torch.nn.Module) -> BasePyTorchClassifier:
        """Wrap a raw nn.Module into BasePyTorchClassifier if needed."""
        if isinstance(model, BasePyTorchClassifier):
            return model
        if isinstance(model, torch.nn.Module):
            return BasePyTorchClassifier(model=model)
        if isinstance(model, BaseModel):
            msg = (
                "ModelTracker requires a BasePyTorchClassifier or torch.nn.Module. "
                f"Received unsupported BaseModel subtype: {type(model)}"
            )
            raise TypeError(msg)
        msg = f"Unsupported model type: {type(model)}"
        raise TypeError(msg)

    @property
    def trackers(self) -> list[Tracker]:
        """Return the list of subscribed trackers."""
        return self._trackers

    def _compute_delta(self, x_adv: torch.Tensor) -> torch.Tensor | None:
        if self._x_orig is None:
            return None
        return x_adv - self._x_orig.to(x_adv.device).detach()

    def _compute_losses(self, scores: torch.Tensor) -> dict[int, torch.Tensor | None]:
        losses: dict[int, torch.Tensor | None] = {}
        y = self._y.to(scores.device) if self._y is not None else None
        for tracker in self._trackers:
            loss = None
            if y is not None:
                tracker_loss_fn = getattr(tracker, "loss_fn", None)
                if tracker_loss_fn is not None:
                    loss = tracker_loss_fn(scores, y)
            losses[id(tracker)] = loss
        return losses

    def _split_trackers(self) -> tuple[list[Tracker], list[Tracker]]:
        grad_trackers = [
            tracker
            for tracker in self._trackers
            if getattr(tracker, "requires_grad", False)
        ]
        non_grad_trackers = [
            tracker
            for tracker in self._trackers
            if not getattr(tracker, "requires_grad", False)
        ]
        return grad_trackers, non_grad_trackers

    def _track_without_grad(
        self,
        trackers: list[Tracker],
        losses: dict[int, torch.Tensor | None],
        scores: torch.Tensor,
        x_adv: torch.Tensor,
        delta: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            for tracker in trackers:
                tracker.track(
                    self._iteration,
                    losses[id(tracker)],
                    scores,
                    x_adv,
                    delta,
                    None,
                )

    def _register_grad_hook(
        self,
        x_input: torch.Tensor,
        trackers: list[Tracker],
        losses: dict[int, torch.Tensor | None],
        scores: torch.Tensor,
        x_adv: torch.Tensor,
        delta: torch.Tensor | None,
    ) -> None:
        if not trackers or not x_input.requires_grad:
            return

        iteration = self._iteration
        hook_handle: torch.utils.hooks.RemovableHandle | None = None

        def _grad_hook(grad: torch.Tensor) -> None:
            nonlocal hook_handle
            if hook_handle is not None:
                hook_handle.remove()
                hook_handle = None
            grad_detached = grad.detach()
            with torch.no_grad():
                for tracker in trackers:
                    tracker.track(
                        iteration,
                        losses[id(tracker)],
                        scores,
                        x_adv,
                        delta,
                        grad_detached,
                    )

        hook_handle = x_input.register_hook(_grad_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not self._tracking or not self._trackers:
            return

        x_input = input[0]
        x_adv = x_input.detach()
        scores = output.detach()
        delta = self._compute_delta(x_adv)
        losses = self._compute_losses(scores)
        grad_trackers, non_grad_trackers = self._split_trackers()

        self._track_without_grad(
            trackers=non_grad_trackers,
            losses=losses,
            scores=scores,
            x_adv=x_adv,
            delta=delta,
        )
        self._register_grad_hook(
            x_input=x_input,
            trackers=grad_trackers,
            losses=losses,
            scores=scores,
            x_adv=x_adv,
            delta=delta,
        )
        self._iteration += 1

    def init_tracking(
        self,
        x_orig: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> None:
        """Initialize tracking for a new batch."""
        self._x_orig = x_orig
        self._y = y
        self._iteration = 0
        self._tracking = True
        for tracker in self._trackers:
            tracker.init_tracking()

    def end_tracking(self) -> None:
        """End tracking for the current batch."""
        self._tracking = False
        for tracker in self._trackers:
            tracker.end_tracking()

    def reset(self) -> None:
        """Reset all subscribed trackers."""
        for tracker in self._trackers:
            tracker.reset()

    def detach(self) -> None:
        """Remove the forward hook from the model."""
        hook_handle = getattr(self, "_hook_handle", None)
        if hook_handle is not None:
            hook_handle.remove()
            self._hook_handle = None

    def __del__(self) -> None:
        """Clean up the forward hook on garbage collection."""
        self.detach()
