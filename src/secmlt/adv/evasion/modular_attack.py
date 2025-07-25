"""Implementation of modular iterative attacks with customizable components."""

from __future__ import annotations  # noqa: I001

from typing import Literal, Union, TYPE_CHECKING
import torch.nn
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import Manipulation
from secmlt.utils.tensor_utils import atleast_kd
from torch.nn import CrossEntropyLoss
from secmlt.optimization.optimizer_factory import OptimizerFactory


if TYPE_CHECKING:
    from functools import partial

    from secmlt.manipulations.manipulation import Manipulation
    from secmlt.models.base_model import BaseModel
    from secmlt.optimization.constraints import Constraint
    from secmlt.optimization.gradient_processing import GradientProcessing
    from secmlt.optimization.initializer import Initializer
    from secmlt.trackers.trackers import Tracker
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
        x_adv, delta = self.manipulation_function(samples, delta)
        x_adv.data, delta.data = self.manipulation_function(samples.data, delta.data)
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
            x_adv.data, delta.data = self.manipulation_function(
                samples.data,
                delta.data,
            )
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
