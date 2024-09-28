"""Ensemble metrics for getting best results across multiple attacks."""

from abc import ABC, abstractmethod
from typing import Union

import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.models.base_model import BaseModel
from secmlt.utils.tensor_utils import atleast_kd
from torch.utils.data import DataLoader, TensorDataset


class Ensemble(ABC):
    """Abstract class for creating an ensemble metric."""

    def __call__(
        self,
        model: BaseModel,
        data_loader: DataLoader,
        adv_loaders: list[DataLoader],
    ) -> DataLoader[torch.Tuple[torch.Tensor]]:
        """
        Get the worst-case of the metric with the given implemented criterion.

        Parameters
        ----------
        model : BaseModel
            Model to use for predictions.
        data_loader : DataLoader
            Test dataloader.
        adv_loaders : list[DataLoader]
            List of dataloaders returned by multiple attacks.

        Returns
        -------
        DataLoader[torch.Tuple[torch.Tensor]]
            The worst-case metric computed on the multiple attacks.
        """
        best_x_adv_data = []
        original_labels = []
        adv_loaders = [iter(a) for a in adv_loaders]
        for samples, labels in data_loader:
            best_x_adv = samples.clone()
            for adv_loader in adv_loaders:
                x_adv, _ = next(adv_loader)
                best_x_adv = self._get_best(model, samples, labels, x_adv, best_x_adv)
            best_x_adv_data.append(best_x_adv)
            original_labels.append(labels)
        best_x_adv_dataset = TensorDataset(
            torch.vstack(best_x_adv_data),
            torch.hstack(original_labels),
        )
        return DataLoader(
            best_x_adv_dataset,
            batch_size=data_loader.batch_size,
        )

    @abstractmethod
    def _get_best(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        x_adv: torch.Tensor,
        best_x_adv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the best result from multiple attacks.

        Parameters
        ----------
        model : BaseModel
            Model to use to predict.
        samples : torch.Tensor
            Input samples.
        labels : torch.Tensor
            Labels for the samples.
        x_adv : torch.Tensor
            Adversarial examples.
        best_x_adv : torch.Tensor
            Best adversarial examples found so far.

        Returns
        -------
        torch.Tensor
            Best adversarial examples between the current x_adv
            and the ones already tested on the given model.
        """
        ...


class MinDistanceEnsemble(Ensemble):
    """Wrapper for ensembling results of multiple minimum-distance attacks."""

    def __init__(self, perturbation_model: str) -> None:
        """
        Create MinDistance Ensemble.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model to use to compute the distance.
        """
        self.perturbation_model = perturbation_model

    def _get_best(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        x_adv: torch.Tensor,
        best_x_adv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the adversarial examples with minimal perturbation.

        Parameters
        ----------
        model : BaseModel
            Model to use to predict.
        samples : torch.Tensor
            Input samples.
        labels : torch.Tensor
            Labels for the samples.
        x_adv : torch.Tensor
            Adversarial examples.
        best_x_adv : torch.Tensor
            Best adversarial examples found so far.

        Returns
        -------
        torch.Tensor
            The minimum-distance adversarial examples found so far.
        """
        preds = model(x_adv).argmax(dim=1)
        is_adv = preds.type(labels.dtype) == labels
        norms = (
            (samples - x_adv)
            .flatten(start_dim=1)
            .norm(LpPerturbationModels.get_p(self.perturbation_model), dim=-1)
        )
        best_adv_norms = (
            (samples - best_x_adv)
            .flatten(start_dim=1)
            .norm(LpPerturbationModels.get_p(self.perturbation_model))
        )
        is_best = torch.logical_and(norms < best_adv_norms, is_adv)

        return torch.where(
            atleast_kd(is_best, len(x_adv.shape)),
            x_adv,
            best_x_adv,
        )


class FixedEpsilonEnsemble(Ensemble):
    """Wrapper for ensembling results of multiple fixed-epsilon attacks."""

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        maximize: bool = True,
        y_target: Union[torch.Tensor, None] = None,
    ) -> None:
        """
        Create fixed epsilon ensemble.

        Parameters
        ----------
        loss_fn : torch.nn.Module
            Loss function to maximize (or minimize).
        maximize : bool, optional
            If True maximizes the loss otherwise it minimizes it, by default True.
        y_target : torch.Tensor | None, optional
            Target label for targeted attacks, None for untargeted, by default None.
        """
        self.maximize = maximize
        self.loss_fn = loss_fn
        self.y_target = y_target

    def _get_best(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        x_adv: torch.Tensor,
        best_x_adv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the adversarial examples with maximum (or minimum) loss.

        Parameters
        ----------
        model : BaseModel
            Model to use to predict.
        samples : torch.Tensor
            Input samples.
        labels : torch.Tensor
            Labels for the samples.
        x_adv : torch.Tensor
            Adversarial examples.
        best_x_adv : torch.Tensor
            Best adversarial examples found so far.

        Returns
        -------
        torch.Tensor
            The maximum-loss adversarial examples found so far.
        """
        if self.y_target is None:
            targets = labels
        else:
            targets = torch.ones_like(labels) * self.y_target
        loss = self.loss_fn(model(x_adv), targets)
        best_adv_loss = self.loss_fn(model(best_x_adv), targets)
        if self.maximize is True:
            is_best = loss > best_adv_loss
        else:
            is_best = loss < best_adv_loss
        return torch.where(
            atleast_kd(is_best, len(x_adv.shape)),
            x_adv,
            best_x_adv,
        )
