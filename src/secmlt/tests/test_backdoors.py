import pytest
import torch
from secmlt.adv.poisoning.base_pytorch_backdoor import BackdoorDatasetPyTorch
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from secmlt.tests.mocks import MockLoss
from torch.optim import SGD


def add_trigger(x: torch.Tensor) -> torch.Tensor:
    return x


@pytest.mark.parametrize(
    ("portion", "poison_indexes"), [(0.1, None), (1.0, None), (None, None), (None, [1])]
)
def test_backdoors(model, dataset, portion, poison_indexes) -> None:
    pytorch_model = model._model
    optimizer = SGD(pytorch_model.parameters(), lr=0.01)
    criterion = MockLoss()

    # create the trainer instance
    trainer = BasePyTorchTrainer(optimizer=optimizer, loss=criterion)

    backdoored_loader = BackdoorDatasetPyTorch(
        dataset,
        trigger_label=0,
        data_manipulation_func=add_trigger,
        portion=portion,
        poisoned_indexes=poison_indexes,
    )
    assert len(backdoored_loader)
    trained_model = trainer.train(pytorch_model, backdoored_loader)
    assert isinstance(trained_model, torch.nn.Module)


@pytest.mark.parametrize(
    ("portion", "poison_indexes"), [(-1.0, None), (2.0, None), (0.5, [1])]
)
def test_backdoors_errors(dataset, portion, poison_indexes) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        BackdoorDatasetPyTorch(
            dataset,
            trigger_label=0,
            data_manipulation_func=add_trigger,
            portion=portion,
            poisoned_indexes=poison_indexes,
        )
