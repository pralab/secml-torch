"""Implementation of min-distance iterative attacks with customizable components."""

from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Union

import torch
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
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler


class ModularEvasionAttackMinDistance(ModularEvasionAttack):
    """Modular evasion attack for min-distance attacks."""

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
        gamma: float = 0.05,
        min_step_size: float | None = None,
        min_gamma: float = 0.001,
        initial_epsilon: float = float("inf"),
    ) -> None:
        """Create a min-distance modular evasion attack.

        Parameters
        ----------
        y_target : int | None
            Target label for the attack, None for untargeted scenarios.
        num_steps : int
            Number of optimisation iterations to perform.
        step_size : float
            Optimiser step size (used as learning rate when the optimiser is
            provided as a string name).
        loss_function : Union[str, torch.nn.Module]
            Loss function to optimise. Strings are resolved via
            ``ModularEvasionAttack``.
        optimizer_cls : str | partial[Optimizer]
            Optimiser factory or name. When a string is provided, a PyTorch
            optimiser will be initialised with ``step_size`` as learning rate.
        scheduler_cls : str | partial[_LRScheduler]
            Scheduler factory or name controlling the optimiser's learning
            rate schedule.
        manipulation_function : Manipulation
            Manipulation applied to the perturbation (e.g., additive with
            domain constraints).
        initializer : Initializer
            Initialiser used to create the starting perturbation.
        gradient_processing : GradientProcessing
            Post-processing applied to gradients before the optimiser step
            (for instance, Lp projection).
        trackers : list[Tracker] | Tracker | None, optional
            Optional tracker(s) collecting metrics during optimisation.
        gamma : float, optional
            Base rate for shrinking/expanding the epsilon ball.
        min_step_size : float | None, optional
            Minimum step size used by cosine annealing schedulers. Defaults to
            ``step_size / 100`` when not specified.
        min_gamma : float, optional
            Lower bound for gamma when subclasses decay it (e.g. cosine decay).
        """
        self.gamma = gamma
        self.min_gamma = min_gamma
        self.min_step_size = (
            min_step_size if min_step_size is not None else step_size / 100
        )
        scheduler_kwargs = {"T_max": num_steps, "eta_min": self.min_step_size}
        self._initial_epsilon = initial_epsilon
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
            scheduler_kwargs=scheduler_kwargs,
        )

    def _preprocess_gradient(
        self,
        grad: torch.Tensor,
        *,
        step: int,
        state: dict,
    ) -> torch.Tensor:
        """Adjust gradients before projection."""
        return grad

    def _init_epsilons(
        self,
        samples: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Initialise epsilon schedule."""
        batch_size = samples.shape[0]
        device, dtype = samples.device, samples.dtype
        return torch.full(
            (batch_size,), self._initial_epsilon, device=device, dtype=dtype
        )

    def _gamma_for_step(self, step: int, state: dict) -> float:
        """Return the gamma value to be used at the given step."""
        return self.gamma

    def _distance_to_boundary(
        self,
        delta: torch.Tensor,
        scores: torch.Tensor,
        target: torch.Tensor,
        state: dict,
    ) -> torch.Tensor | None:
        """Return the distance-to-boundary term when epsilons expand."""
        return None

    def _update_epsilons(
        self,
        *,
        step: int,
        delta: torch.Tensor,
        is_adv: torch.Tensor,
        best_distances: torch.Tensor,
        adv_found: torch.Tensor,
        scores: torch.Tensor,
        target: torch.Tensor,
        epsilons: torch.Tensor,
        state: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Update epsilons and any custom state after each optimisation step."""
        gamma_value = self._gamma_for_step(step, state)

        if self.perturbation_model == 0:
            epsilons = torch.where(
                is_adv,
                torch.minimum(
                    epsilons - 1,
                    torch.minimum(
                        best_distances, torch.floor(epsilons * (1 - gamma_value))
                    ),
                ),
                torch.maximum(torch.floor(epsilons * (1 + gamma_value)), epsilons + 1),
            )
        else:
            distance = self._distance_to_boundary(
                delta=delta,
                scores=scores,
                target=target,
                state=state,
            )
            if distance is not None:
                growth_if_no_success = best_distances + distance
            else:
                growth_if_no_success = epsilons * (1 + gamma_value)

            epsilons = torch.where(
                is_adv,
                torch.minimum(best_distances, epsilons * (1 - gamma_value)),
                torch.where(
                    adv_found,
                    epsilons * (1 + gamma_value),
                    growth_if_no_success,
                ),
            )

        epsilons = torch.clamp(epsilons, min=0.0)
        return epsilons, state

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
        x_adv, delta = self.manipulation_function(samples, delta)
        device, dtype = samples.device, samples.dtype
        batch_size = samples.shape[0]

        best_distances = torch.full(
            (batch_size,), float("inf"), device=device, dtype=dtype
        )
        best_delta = torch.zeros_like(samples)
        best_adv = samples.detach().clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

        epsilons = self._init_epsilons(samples, delta)
        state: dict = {}
        self.manipulation_function.perturbation_constraints[0].radius = epsilons

        for step in range(self.num_steps):
            x_adv.data, delta.data = self.manipulation_function(
                samples.data,
                delta.data,
            )
            delta_before_processing = delta.detach().clone()
            optimizer.zero_grad()

            scores, losses = self._loss_and_grad(
                model=model,
                samples=samples,
                delta=delta,
                target=target,
                multiplier=multiplier,
            )

            predictions = scores.argmax(dim=1)
            is_adv = predictions == target if multiplier == 1 else predictions != target

            distances = (
                delta.detach()
                .flatten(start_dim=1)
                .norm(
                    p=self.perturbation_model,
                    dim=-1,
                )
            )
            improvement = torch.logical_and(is_adv, distances < best_distances)
            mask = atleast_kd(improvement, delta.ndim)

            best_delta.data = torch.where(
                mask,
                delta_before_processing.data,
                best_delta.data,
            )
            best_adv.data = torch.where(
                mask,
                x_adv.detach(),
                best_adv.data,
            )
            best_distances.data = torch.where(
                improvement,
                distances,
                best_distances,
            )

            adv_found.logical_or_(is_adv)

            grad_before_processing = delta.grad.detach().clone()
            grad_to_project = self._preprocess_gradient(
                grad_before_processing.clone(),
                step=step,
                state=state,
            )
            delta.grad.data = self.gradient_processing(grad_to_project)

            optimizer.step()
            scheduler.step()

            if self.trackers is not None:
                trackers = self.trackers
                if isinstance(trackers, list):
                    for tracker in trackers:
                        tracker.track(
                            step,
                            losses.detach().cpu().data,
                            scores.detach().cpu().data,
                            x_adv.detach().cpu().data,
                            delta.detach().cpu().data,
                            grad_before_processing.detach().cpu().data,
                        )
                else:
                    trackers.track(
                        step,
                        losses.detach().cpu().data,
                        scores.detach().cpu().data,
                        x_adv.detach().cpu().data,
                        delta.detach().cpu().data,
                        grad_before_processing.detach().cpu().data,
                    )

            epsilons, state = self._update_epsilons(
                step=step,
                delta=delta,
                is_adv=is_adv,
                best_distances=best_distances,
                adv_found=adv_found,
                scores=scores,
                target=target,
                epsilons=epsilons,
                state=state,
            )

            self.manipulation_function.perturbation_constraints[0].radius = epsilons

        self.manipulation_function.perturbation_constraints[0].radius = best_distances
        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta
