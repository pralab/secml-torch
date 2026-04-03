import torch
from secmlt.metrics.classification import (
    Accuracy,
    AccuracyEnsemble,
    AttackSuccessRate,
    EnsembleSuccessRate,
)
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader, TensorDataset


def test_accuracy(model, data_loader) -> None:
    acc_metric = Accuracy()
    acc = acc_metric(model, data_loader)
    assert torch.is_tensor(acc)


def test_attack_success_rate(model, adv_loaders):
    attack_acc = AttackSuccessRate()
    acc = attack_acc(model, adv_loaders[0])
    assert torch.is_tensor(acc)


def test_accuracy_ensemble(model, adv_loaders):
    acc_ensemble = AccuracyEnsemble()
    acc = acc_ensemble(model, adv_loaders)
    assert torch.is_tensor(acc)


def test_ensemble_success_rate(model, adv_loaders):
    ensemble_acc = EnsembleSuccessRate()
    acc = ensemble_acc(model, adv_loaders)
    assert torch.is_tensor(acc)


def test_accuracy_wraps_raw_nn_module(data_loader):
    model = torch.nn.Linear(32 * 32 * 3, 10)

    class FlattenModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            return self.base_model(x.view(x.shape[0], -1))

    acc_metric = Accuracy()
    acc = acc_metric(FlattenModel(model), data_loader)

    assert torch.is_tensor(acc)


def test_accuracy_uses_predict_not_call(data_loader):
    class PredictOnlyModel(BaseModel):
        def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.zeros(x.shape[0], dtype=torch.long)

        def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.zeros(x.shape[0], 10)

        def train(self, dataloader):
            return self

        def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            msg = "__call__ should not be used by Accuracy"
            raise AssertionError(msg)

    acc_metric = Accuracy()
    acc = acc_metric(PredictOnlyModel(), data_loader)

    assert torch.is_tensor(acc)


def test_accuracy_ensemble_uses_predict_not_call(data_loader):
    class PredictOnlyModel(BaseModel):
        def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.ones(x.shape[0], dtype=torch.long)

        def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.zeros(x.shape[0], 10)

        def train(self, dataloader):
            return self

        def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            msg = "__call__ should not be used by AccuracyEnsemble"
            raise AssertionError(msg)

    xs = torch.randn(4, 3, 32, 32)
    ys = torch.randint(0, 10, (4,))
    loader_a = DataLoader(TensorDataset(xs, ys), batch_size=2)
    loader_b = DataLoader(TensorDataset(xs + 0.01, ys), batch_size=2)

    acc_metric = AccuracyEnsemble()
    acc = acc_metric(PredictOnlyModel(), [loader_a, loader_b])

    assert torch.is_tensor(acc)
