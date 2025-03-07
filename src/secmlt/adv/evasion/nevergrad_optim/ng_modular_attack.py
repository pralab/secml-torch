"""Modular iterative attacks with customizable components with nevergrad."""

import nevergrad
import torch
from nevergrad.parametrization.core import Parameter
from secmlt.adv.evasion.modular_attack import ModularEvasionAttackFixedEps
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import ClipConstraint
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class NgModularEvasionAttackFixedEps(ModularEvasionAttackFixedEps):
    """Modular evasion attack for fixed-epsilon attacks, using nevergrad as backend."""

    def _optimizer_step(
            self, optimizer: nevergrad.optimization.base.Optimizer,
            delta: nevergrad.p.Array,
            loss: torch.Tensor
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
        optimizer.tell(delta, loss.item())
        return optimizer.ask()

    def _apply_manipulation(
            self, x: torch.Tensor, delta: nevergrad.p.Array
    ) -> (torch.Tensor, torch.Tensor):
        p_delta = torch.from_numpy(delta.value)
        return self.manipulation_function(x.data, p_delta)

    def _create_optimizer(self, delta: nevergrad.p.Array, **kwargs) -> Optimizer:
        constraints = self.manipulation_function.domain_constraints
        upper, lower = 1, 0
        if constraints is not None:
            for constraint in constraints:
                if isinstance(constraint, ClipConstraint):
                    upper, lower = max(upper, constraint.ub), min(lower, constraint.lb)
        return self.optimizer_cls(
            parametrization=nevergrad.p.Array(
                shape=delta.value.shape, lower=lower, upper=upper)
        )

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
            Dataloader with adversarial examples and original labels.
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
        original_labels = torch.vstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        return DataLoader(
            adversarial_dataset,
            batch_size=data_loader.batch_size,
        )
