from abc import ABC, abstractmethod
from secmlt.adv.evasion.perturbation_models import PerturbationModels
from secmlt.utils.tensor_utils import atleast_kd
import torch
from torch.utils.data import DataLoader, TensorDataset


class Ensemble(ABC):
    def __call__(self, model, data_loader, adv_loaders):
        best_x_adv_data = []
        original_labels = []
        adv_loaders = [iter(a) for a in adv_loaders]
        for samples, labels in data_loader:
            best_x_adv = samples.clone()
            for adv_loader in adv_loaders:
                x_adv, _ = next(adv_loader)
                best_x_adv = self.get_best(model, samples, labels, x_adv, best_x_adv)
            best_x_adv_data.append(best_x_adv)
            original_labels.append(labels)
        best_x_adv_dataset = TensorDataset(
            torch.vstack(best_x_adv_data), torch.hstack(original_labels)
        )
        best_x_adv_loader = DataLoader(
            best_x_adv_dataset, batch_size=data_loader.batch_size
        )
        return best_x_adv_loader

    @abstractmethod
    def get_best(self, model, samples, labels, x_adv): ...


class MinDistanceEnsemble(Ensemble):
    def get_best(self, model, samples, labels, x_adv, best_x_adv):
        preds = model(x_adv).argmax(dim=1)
        is_adv = preds.type(labels.dtype) == labels
        norms = (
            (samples - x_adv)
            .flatten(start_dim=1)
            .norm(PerturbationModels.get_p(self.perturbation_model), dim=-1)
        )
        best_adv_norms = (
            (samples - best_x_adv)
            .flatten(start_dim=1)
            .norm(PerturbationModels.get_p(self.perturbation_model))
        )
        is_best = torch.logical_and(norms < best_adv_norms, is_adv)
        best_x_adv = torch.where(
            atleast_kd(is_best, len(x_adv.shape)), x_adv, best_x_adv
        )
        return best_x_adv


class FixedEpsilonEnsemble(Ensemble):
    def __init__(self, loss_fn, maximize=True, y_target=None) -> None:
        self.maximize = maximize
        self.loss_fn = loss_fn
        self.y_target = y_target

    def get_best(self, model, samples, labels, x_adv, best_x_adv):
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
        best_x_adv = torch.where(
            atleast_kd(is_best, len(x_adv.shape)), x_adv, best_x_adv
        )
        return best_x_adv
