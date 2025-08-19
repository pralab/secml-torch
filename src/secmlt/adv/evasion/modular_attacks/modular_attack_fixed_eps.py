"""Implementation of fixed-epsilon iterative attacks with customizable components."""

from __future__ import annotations  # noqa: I001

from typing import Union, TYPE_CHECKING
import torch.nn
from secmlt.adv.evasion.modular_attacks.modular_attack import ModularEvasionAttack
from secmlt.manipulations.manipulation import Manipulation
from secmlt.utils.tensor_utils import atleast_kd


if TYPE_CHECKING:
    from functools import partial

    from secmlt.manipulations.manipulation import Manipulation
    from secmlt.models.base_model import BaseModel
    from secmlt.optimization.gradient_processing import GradientProcessing
    from secmlt.optimization.initializer import Initializer
    from secmlt.trackers.trackers import Tracker
    from torch.optim import LRScheduler, Optimizer


class ModularEvasionAttackFixedEps(ModularEvasionAttack):
    """Modular evasion attack for fixed-epsilon attacks."""

    def __init__(
        self,
        y_target: int | None,
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: str | partial[Optimizer],
        scheduler_cls: str | partial[LRScheduler],
        manipulation_function: Manipulation,
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: list[Tracker] | Tracker | None = None,
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
            Step size for the attack.
        loss_function : Union[str, torch.nn.Module]
            Loss function to be used in the attack.
        optimizer_cls : str | partial[Optimizer]
            Optimizer class or partial function to create the optimizer.
        scheduler_cls : str | partial[LRScheduler]
            Scheduler class or partial function to create the scheduler.
        manipulation_function : Manipulation
            Function to manipulate the input data.
        initializer : Initializer
            Initializer for the perturbation.
        gradient_processing : GradientProcessing
            Gradient processing method.
        trackers : list[Tracker] | Tracker | None, optional
            Trackers for monitoring the attack, by default None.
        """
        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            gradient_processing=gradient_processing,
            trackers=trackers,
        )

    def _run_loop(
        self,
        model: BaseModel,
        delta: torch.Tensor,
        samples: torch.Tensor,
        target: torch.Tensor,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        multiplier: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_adv, delta = self.manipulation_function(samples, delta)
        best_losses = torch.zeros(samples.shape[0]).fill_(torch.inf)
        best_delta = torch.zeros_like(samples)

        for i in range(self.num_steps):
            scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
            losses *= multiplier
            loss = losses.sum()
            optimizer.zero_grad()
            loss.backward()
            grad_before_processing = delta.grad.data
            delta.grad.data = self.gradient_processing(delta.grad.data)
            optimizer.step()
            scheduler.step()
            x_adv.data, delta.data = self.manipulation_function(
                samples.data,
                delta.data,
            )
            if self.trackers is not None:
                if isinstance(self.trackers, list):
                    for tracker in self.trackers:
                        tracker.track(
                            i,
                            losses.detach().cpu().data,
                            scores.detach().cpu().data,
                            x_adv.detach().cpu().data,
                            delta.detach().cpu().data,
                            grad_before_processing.detach().cpu().data,
                        )
                else:
                    self.trackers.track(
                        i,
                        losses.detach().cpu().data,
                        scores.detach().cpu().data,
                        x_adv.detach().cpu().data,
                        delta.detach().cpu().data,
                        grad_before_processing.detach().cpu().data,
                    )

            # keep perturbation with highest loss
            best_delta.data = torch.where(
                atleast_kd(losses.detach().cpu() < best_losses, len(samples.shape)),
                delta.data,
                best_delta.data,
            )
            best_losses.data = torch.where(
                losses.detach().cpu() < best_losses,
                losses.detach().cpu(),
                best_losses.data,
            )
        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta