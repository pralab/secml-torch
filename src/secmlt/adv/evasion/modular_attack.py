"""Implementation of modular iterative attacks with customizable components."""

from functools import partial
from typing import Literal, Optional, Union

import torch.nn
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import Manipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import Constraint
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.trackers.trackers import Tracker
from secmlt.utils.tensor_utils import atleast_kd
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

CE_LOSS = "ce_loss"
LOGIT_LOSS = "logit_loss"

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
}


class ModularEvasionAttackFixedEps(BaseEvasionAttack):
    """Modular evasion attack for fixed-epsilon attacks."""

    def __init__(
            self,
            y_target: int | None,
            num_steps: int,
            step_size: float,
            loss_function: Union[str, torch.nn.Module],
            optimizer_cls: str | partial[Optimizer],
            manipulation_function: Manipulation,
            initializer: Initializer,
            gradient_processing: GradientProcessing,
            budget: Optional[int],
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
            Attack step size.
        loss_function : str | torch.nn.Module
            Loss function to minimize.
        optimizer_cls : str | partial[Optimizer]
            Algorithm for solving the attack optimization problem.
        manipulation_function : Manipulation
            Manipulation function to perturb the inputs.
        initializer : Initializer
            Initialization for the perturbation delta.
        gradient_processing : GradientProcessing
            Gradient transformation function.
        budget: int
            The maximum allowed number of queries (forward + backward).
            If left to None, it will be double the num_steps.
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
        self.trackers = trackers
        self.budget = budget
        if self.budget is None:
            self.budget = 2 * num_steps
        if isinstance(loss_function, str):
            if loss_function in LOSS_FUNCTIONS:
                self.loss_function = LOSS_FUNCTIONS[loss_function](reduction="none")
            else:
                msg = (
                    f"Loss function not found. Use one among {LOSS_FUNCTIONS.values()}"
                )
                raise ValueError(msg)
        else:
            self.loss_function = loss_function

        if isinstance(optimizer_cls, str):
            optimizer_cls = OptimizerFactory.create_from_name(
                optimizer_cls,
                lr=step_size,
            )

        self.optimizer_cls = optimizer_cls

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

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        """
        Check if a given perturbation model is implemented.

        Returns
        -------
        set[str]
            Set of perturbation models available for this attack.
        """
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }

    @classmethod
    def _trackers_allowed(cls) -> Literal[True]:
        return True

    def _init_perturbation_constraints(self) -> list[Constraint]:
        msg = "Must be implemented accordingly"
        raise NotImplementedError(msg)

    def _create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
        return self.optimizer_cls([delta], lr=self.step_size, **kwargs)

    @classmethod
    def consumed_budget(cls) -> int:
        """
        Return the amount of budget needed at each iteration.

        Returns
        -------
        int
            The amount of queries (forward + backward).
        """
        return 2

    def optimizer_step(self, optimizer: Optimizer, delta: torch.Tensor,
                       loss: torch.Tensor) -> torch.Tensor:
        """
        Perform the optimization step to optimize the attack.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer used by the attack
        delta : torch.Tensor
            The manipulation computed by the attack
        loss : torch.Tensor
            The loss computed by the attack for the provided delta

        Returns
        -------
        torch.Tensor
            Resulting manipulation after the optimization step
        """
        optimizer.zero_grad()
        loss.backward()
        delta.grad.data = self.gradient_processing(delta.grad.data)
        optimizer.step()
        return delta

    def apply_manipulation(
            self, x: torch.Tensor, delta: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Apply the manipulation during the attack.

        Parameters
        ----------
        x : torch.Tensor
            input sample to manipulate
        delta : torch.Tensor
            manipulation to apply

        Returns
        -------
        torch.Tensor, torch.Tensor
            the manipulated sample and the manipulation itself
        """
        return self.manipulation_function(x, delta)

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

    def _run(
            self,
            model: BaseModel,
            samples: torch.Tensor,
            labels: torch.Tensor,
            init_deltas: torch.Tensor = None,
            optim_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if optim_kwargs is None:
            optim_kwargs = {}
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

        optimizer = self._create_optimizer(delta, **optim_kwargs)
        x_adv, delta = self.apply_manipulation(samples, delta)
        best_losses = torch.zeros(samples.shape[0]).fill_(torch.inf)
        best_delta = torch.zeros_like(samples)
        available_budget = self.budget
        for i in range(self.num_steps):
            scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
            losses *= multiplier
            loss = losses.sum()
            delta = self.optimizer_step(optimizer, delta, loss)
            x_adv.data, delta = self.apply_manipulation(
                samples.data,
                delta,
            )
            self._track(delta, i, losses, scores, x_adv)
            self._set_best_results(best_delta, best_losses, delta, losses, samples)
            available_budget -= self.consumed_budget()
            if available_budget <= 0:
                break
        x_adv, _ = self.apply_manipulation(samples.data, best_delta)
        return x_adv, best_delta

    def _set_best_results(self,
                          best_delta: torch.Tensor,
                          best_losses: torch.Tensor,
                          delta: torch.Tensor,
                          losses: torch.Tensor,
                          samples: torch.Tensor) -> None:
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

    def _track(self, delta: torch.Tensor,
               i: int,
               losses: torch.Tensor,
               scores: torch.Tensor,
               x_adv: torch.Tensor) -> None:
        if self.trackers is not None:
            for tracker in self.trackers:
                tracker.track(
                    i,
                    losses.detach().cpu().data,
                    scores.detach().cpu().data,
                    x_adv.detach().cpu().data,
                    delta.detach().cpu().data,
                    delta.grad.detach().cpu().data,
                )
