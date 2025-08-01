"""Implementation of min-distance iterative attacks with customizable components."""

from __future__ import annotations  # noqa: I001

import math
from typing import Union, TYPE_CHECKING
from secmlt.optimization.losses import LogitDifferenceLoss
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


class ModularEvasionAttackMinDistance(ModularEvasionAttack):
    """Modular evasion attack for min-distance attacks."""

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
        gamma: float = 0.05,
        min_step_size: float | None = None,
        min_gamma: float = 0.001,
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
        self.gamma = gamma
        self.min_gamma = min_gamma
        self.min_step_size = (
            min_step_size if min_step_size is not None else step_size / 100
        )
        scheduler_kwargs = {"T_max": num_steps, "eta_min": self.min_step_size}
        optimizer_kwargs = None
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
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
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
        best_distances = torch.zeros(samples.shape[0]).fill_(torch.inf)
        best_delta = torch.zeros_like(samples)
        epsilons = torch.full((x_adv.shape[0],), torch.inf, device=x_adv.device)
        gamma = self.gamma
        adv_found = torch.zeros(samples.shape[0], dtype=torch.bool, device=x_adv.device)

        for i in range(self.num_steps):
            scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
            is_adv = (
                scores.argmax(dim=1) == target
                if multiplier == 1
                else scores.argmax(dim=1) != target
            )
            distances = torch.norm(
                delta.detach().cpu().flatten(start_dim=1),
                p=self.perturbation_model,
                dim=-1,
            )
            condition = torch.logical_and(
                is_adv,
                distances.detach() < best_distances,
            )

            # keep perturbation with smallest distance and adv
            best_delta.data = torch.where(
                atleast_kd(
                    condition,
                    k=len(delta.shape),
                ),
                delta.data,
                best_delta.data,
            )

            # save best distances found for successful adv
            best_distances.data = torch.where(
                condition,
                distances,
                best_distances,
            )
            x_adv.data, delta.data = self.manipulation_function(
                samples.data,
                delta.data,
            )
            losses *= multiplier
            loss = losses.sum()
            optimizer.zero_grad()
            loss.backward()
            grad_before_processing = delta.grad.data
            delta.grad.data = self.gradient_processing(delta.grad.data)
            optimizer.step()
            scheduler.step(epoch=i)

            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(
                        i,
                        losses.detach().cpu().data,
                        scores.detach().cpu().data,
                        x_adv.detach().cpu().data,
                        delta.detach().cpu().data,
                        grad_before_processing.detach().cpu().data,
                    )

            adv_found = torch.logical_or(adv_found, is_adv)

            # update epsilons
            if self.perturbation_model == 0:
                epsilons = torch.where(
                    is_adv,
                    torch.minimum(
                        epsilons - 1,
                        torch.minimum(
                            best_distances, torch.floor(epsilons * (1 - gamma))
                        ),
                    ),
                    torch.maximum(torch.floor(epsilons * (1 + gamma)), epsilons + 1),
                )
            else:
                logits_difference_loss = LogitDifferenceLoss()(scores, target)
                distance_to_boundary = logits_difference_loss / delta.data.flatten(
                    start_dim=1
                ).norm(p=self.perturbation_model_dual, dim=-1)
                epsilons = torch.where(
                    is_adv,
                    torch.minimum(best_distances, epsilons * (1 - gamma)),
                    torch.where(
                        adv_found,
                        epsilons * (1 + gamma),
                        best_distances + distance_to_boundary,
                    ),
                )

            epsilons = torch.clamp(epsilons, 0)

            # cosine annealing for gamma
            gamma = (
                self.min_gamma
                + (self.gamma - self.min_gamma)
                * (1 + math.cos(math.pi * i / self.num_steps))
                / 2
            )
            self.manipulation_function.perturbation_constraints[0].radius = epsilons

        # do not apply constraint on last operation
        self.manipulation_function.perturbation_constraints[0].radius = torch.inf
        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta
