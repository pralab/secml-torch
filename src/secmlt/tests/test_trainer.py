import torch
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from secmlt.tests.mocks import MockLoss
from torch.optim import SGD


def test_pytorch_trainer(model, data_loader) -> None:
    pytorch_model = model._model
    optimizer = SGD(pytorch_model.parameters(), lr=0.01)
    criterion = MockLoss()

    # Create the trainer instance
    trainer = BasePyTorchTrainer(optimizer=optimizer, loss=criterion)

    # Train the model
    trained_model = trainer.train(pytorch_model, data_loader)
    assert isinstance(trained_model, torch.nn.Module)
