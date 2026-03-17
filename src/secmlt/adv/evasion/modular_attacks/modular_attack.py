"""Implementation of modular iterative attacks with customizable components."""

from __future__ import annotations  # noqa: I001

from abc import abstractmethod
from typing import Literal, Union, TYPE_CHECKING
import torch.nn
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.manipulations.manipulation import Manipulation
from torch.nn import CrossEntropyLoss
from secmlt.optimization.losses import LogitDifferenceLoss
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory
from secmlt.trackers.trackers import Tracker


if TYPE_CHECKING:
    from functools import partial

    from secmlt.manipulations.manipulation import Manipulation
    from secmlt.models.base_model import BaseModel
    from secmlt.optimization.constraints import Constraint
    from secmlt.optimization.gradient_processing import GradientProcessing
    from secmlt.optimization.initializer import Initializer
    from torch.optim import Optimizer, _LRScheduler

CE_LOSS = "ce_loss"
LOGIT_LOSS = "logit_loss"

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
    LOGIT_LOSS: LogitDifferenceLoss,
}


class ModularEvasionAttack(BaseEvasionAttack):
    """Modular evasion attack."""

    def __init__(
        self,
        y_target: int | None,
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: str | partial[Optimizer],
        scheduler_cls: str | partial[_LRScheduler],
        manipulation_function: Manipulation,
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: list[Tracker] | Tracker | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        """
        Create modular evasion attack.

        Parameters
        ----------
        y_target : int | None
            Target label for the attack, None for untargeted.
        num_steps : int
            Number of iterations for the attack.
        step_size : float
            Attack step size.
        loss_function : str | torch.nn.Module
            Loss function to minimize.
        optimizer_cls : str | partial[Optimizer]
            Algorithm for solving the attack optimization problem.
        scheduler_cls : str | partial[LRScheduler]
            Learning rate scheduler for the optimizer.
        manipulation_function : Manipulation
            Manipulation function to perturb the inputs.
        initializer : Initializer
            Initialization for the perturbation delta.
        gradient_processing : GradientProcessing
            Gradient transformation function.
        trackers : list[Tracker] | Tracker | None, optional
            Trackers for logging, by default None.

        Raises
        ------
        ValueError
            Raises ValueError if the loss is not in allowed
            list of loss functions.
        """
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size
        if isinstance(trackers, Tracker):
            self.trackers = [trackers]
        else:
            self.trackers = trackers
        if isinstance(loss_function, str):
            if loss_function in LOSS_FUNCTIONS:
                self.loss_function = LOSS_FUNCTIONS[loss_function](reduction="none")
            else:
                msg = (
                    f"Loss function not found. Use one among {LOSS_FUNCTIONS.values()}"
                )
                raise ValueError(msg)
        else:
            self._loss_function = loss_function

        if isinstance(optimizer_cls, str):
            optimizer_cls = OptimizerFactory.create_from_name(
                optimizer_cls,
                lr=step_size,
            )

        if isinstance(scheduler_cls, str):
            scheduler_cls = LRSchedulerFactory.create_scheduler_from_name(
                scheduler_cls,
                optimizer_cls=optimizer_cls,
            )

        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls

        self.optim_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else {}

        self._manipulation_function = manipulation_function
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__()

    @property
    def manipulation_function(self) -> Manipulation:
        """
        Get the manipulation function for the attack.

        Returns
        -------
        Manipulation
            The manipulation function used in the attack.
        """
        return self._manipulation_function

    @manipulation_function.setter
    def manipulation_function(self, manipulation_function: Manipulation) -> None:
        """
        Set the manipulation function for the attack.

        Parameters
        ----------
        manipulation_function : Manipulation
            The manipulation function to be used in the attack.
        """
        self._manipulation_function = manipulation_function

    @property
    def loss_function(self) -> torch.nn.Module:
        """Get the loss function of the attack."""
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: torch.nn.Module) -> None:
        """Set the loss function of the attack."""
        self._loss_function = loss_function

    @classmethod
    def _trackers_allowed(cls) -> Literal[True]:
        return True

    def _init_perturbation_constraints(self) -> list[Constraint]:
        msg = "Must be implemented accordingly"
        raise NotImplementedError(msg)

    def _create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
        return self.optimizer_cls([delta], lr=self.step_size, **kwargs)

    def _create_scheduler(self, optimizer: Optimizer, **kwargs) -> _LRScheduler:
        return self.scheduler_cls(optimizer, **kwargs)

    def forward_loss(
        self, model: BaseModel, x: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward for the loss function.

        Parameters
        ----------
        model : BaseModel
            Model used by the attack run.
        x : torch.Tensor
            Input sample.
        target : torch.Tensor
            Target for computing the loss.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output scores and loss.
        """
        scores = model.decision_function(x)
        target = target.to(scores.device)
        losses = self.loss_function(scores, target)
        return scores, losses

    def _loss_and_grad(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        delta: torch.Tensor,
        target: torch.Tensor,
        multiplier: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores and losses, then backward to get delta.grad.

        Compute scores and per-sample losses for x_adv,
        then backward the summed loss to populate delta.grad (gradient w.r.t. delta).
        Returns (scores, losses) where losses is per-sample (detached).
        Assumes optimizer.zero_grad() has been called outside.
        """
        # Ensure delta requires grad and is leaf so autograd will populate delta.grad
        # (If delta is not a leaf, set requires_grad on a clone and operate on that.)
        if not delta.requires_grad:
            delta.requires_grad_()

        # Ensure we start with zeroed grads
        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        # Build the adversarial example from samples and delta
        x_adv, _ = self.manipulation_function(
            samples, delta
        )  # must be a function of delta

        # Forward: get scores and per-sample losses
        scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
        losses = losses * multiplier  # keep same convention as before

        # Backward on summed loss -> populates delta.grad
        loss_sum = losses.sum()
        loss_sum.backward()

        # At this point delta.grad is the gradient of loss_sum w.r.t. delta
        return scores, losses.detach()

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)

        if init_deltas is not None:
            delta = init_deltas.data
        elif isinstance(self.initializer, BaseEvasionAttack):
            _, delta = self.initializer._run(model, samples, target)
        else:
            delta = self.initializer(samples.data)
        delta.requires_grad = True

        optimizer = self._create_optimizer(delta, **self.optim_kwargs)
        scheduler = self._create_scheduler(optimizer, **self.scheduler_kwargs)
        return self._run_loop(
            model,
            delta,
            samples,
            target,
            optimizer,
            scheduler,
            multiplier,
        )

    @abstractmethod
    def _run_loop(
        self,
        model: BaseModel,
        delta: torch.Tensor,
        samples: torch.Tensor,
        target: torch.Tensor,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        multiplier: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Abstract run loop for the attack."""
        raise NotImplementedError
