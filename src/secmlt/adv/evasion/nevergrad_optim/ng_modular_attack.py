"""Modular iterative attacks with customizable components with nevergrad."""

from functools import partial
from typing import Literal, Optional, Union

import nevergrad
import numpy as np
import torch
from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.parametrization.core import Parameter
from secmlt.adv.evasion.modular_attack import ModularEvasionAttackFixedEps
from secmlt.manipulations.manipulation import Manipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import ClipConstraint
from secmlt.optimization.gradient_processing import NoGradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.trackers import Tracker
from secmlt.utils.tensor_utils import atleast_kd
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class NgModularEvasionAttackFixedEps(ModularEvasionAttackFixedEps):
    """Modular evasion attack for fixed-epsilon attacks, using nevergrad as backend."""

    def __init__(
        self,
        y_target: int | None,
        num_steps: int,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: str | partial[ConfiguredOptimizer] | ConfiguredOptimizer,
        manipulation_function: Manipulation,
        initializer: Initializer,
        budget: Optional[int],
        trackers: list[Tracker] | Tracker | None = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Create the generic modular attack using an optimizer from nevergrad.

        Parameters
        ----------
         y_target : int | None, optional
            Target label for a targeted attack, None
            for untargeted attack, by default None.
        num_steps : int
            number of maximum steps
        loss_function : Union[str, torch.nn.Module]
            a Pytorch loss function, specified as string or object
        optimizer_cls : str | partial[ConfiguredOptimizer]
            the optimizer from nevergrad, specified as string or class
        manipulation_function : Manipulation
            the type of manipulation to apply
        initializer: Initializer
            the type of initialization for the manipulation
        budget: Optional[int]
            the amount of the query budget of the attack
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            available only for native implementation, by default None.
        random_state: Optional[int]
            set the random seed of the nevergrad algorithm.
            Set None to keep randomness.
        """
        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            gradient_processing=NoGradientProcessing(),
            budget=budget,
            trackers=trackers,
            step_size=0,
        )
        self.random_state = random_state

    @classmethod
    def _trackers_allowed(cls) -> Literal[False]:
        return False

    def optimizer_step(
        self,
        optimizer: nevergrad.optimization.base.Optimizer,
        delta: nevergrad.p.Array,
        loss: torch.Tensor,
    ) -> Parameter:
        """
        Perform the optimization step to optimize the attack.

        Parameters
        ----------
        optimizer : nevergrad.optimization.base.Optimizer
            The optimizer used by the attack
        delta : nevergrad.p.Array
            The manipulation computed by the attack
        loss : torch.Tensor
            The loss computed by the attack for the provided delta

        Returns
        -------
        nevergrad.p.Array
            Resulting manipulation after the optimization step
        """
        if isinstance(delta, torch.Tensor):
            delta = optimizer.parametrization.spawn_child(new_value=delta.numpy())
        optimizer.tell(delta, loss.item())
        return optimizer.ask()

    def apply_manipulation(
        self, x: torch.Tensor, delta: nevergrad.p.Array
    ) -> (torch.Tensor, torch.Tensor):
        """
        Apply the manipulation during the attack.

        Parameters
        ----------
        x : torch.Tensor
            input sample to manipulate
        delta : nevergrad.p.Array
            manipulation to apply

        Returns
        -------
        torch.Tensor, torch.Tensor
            the manipulated sample and the manipulation itself
        """
        if not isinstance(delta, torch.Tensor):
            p_delta = torch.from_numpy(delta.value).float()
        else:
            p_delta = delta.data
        x_adv, proj_delta = self.manipulation_function(x.data, p_delta)
        return x_adv, proj_delta

    def _create_optimizer(self, delta: nevergrad.p.Array, **kwargs) -> Optimizer:
        constraints = self.manipulation_function.domain_constraints
        upper, lower = 1, 0
        if constraints is not None:
            for constraint in constraints:
                if isinstance(constraint, ClipConstraint):
                    upper, lower = max(upper, constraint.ub), -max(upper, constraint.ub)
        optimizer = self.optimizer_cls(
            parametrization=nevergrad.p.Array(
                shape=delta.value.shape, lower=lower, upper=upper
            ),
        )
        if self.random_state is not None:
            random_state = np.random.RandomState(self.random_state)
            optimizer.random_state = random_state
            optimizer.parametrization.random_state = random_state
        return optimizer

    def _set_best_results(
        self,
        best_delta: torch.Tensor,
        best_losses: torch.Tensor,
        delta: nevergrad.p.Array,
        losses: torch.Tensor,
        samples: torch.Tensor,
    ) -> None:
        best_delta.data = torch.where(
            atleast_kd(losses.detach().cpu() < best_losses, len(samples.shape)),
            delta,
            best_delta,
        )
        best_losses.data = torch.where(
            losses.detach().cpu() < best_losses,
            losses.detach().cpu(),
            best_losses.data,
        )

    @classmethod
    def consumed_budget(cls) -> int:
        """
        Return the amount of budget consumed by the attack.

        Returns
        -------
        int
            The amount of queries (forward + backward).
        """
        return 1

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        """
        Compute the attack against the model, using the input data.

        Parameters
        ----------
        model : BaseModel
            Model to test.
        data_loader : DataLoader
            Test dataloader.

        Returns
        -------
        DataLoader
            Data loader with adversarial examples and original labels.
        """
        adversarials = []
        original_labels = []
        for samples, labels in data_loader:
            for sample, label in zip(samples, labels, strict=False):
                sample = sample.unsqueeze(0)
                label = label.unsqueeze(0)
                x_adv, _ = self._run(model, sample, label)
                adversarials.append(x_adv)
                original_labels.append(label)
        adversarials = torch.vstack(adversarials)
        original_labels = torch.vstack(original_labels).flatten()
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        return DataLoader(
            adversarial_dataset,
            batch_size=data_loader.batch_size,
        )
