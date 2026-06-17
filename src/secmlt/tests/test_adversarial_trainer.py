import pytest
import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGDNative
from secmlt.defenses.adv_training.pytorch.adversarial_trainer import AdversarialTrainer
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.tests.mocks import MockLoss, MockModel
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset


def _make_attack() -> PGDNative:
    return PGDNative(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.3,
        num_steps=3,
        step_size=0.1,
        random_start=False,
        y_target=None,
    )


def test_adversarial_trainer_with_module() -> None:
    # 1. Create random dataset
    data = torch.randn(20, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 9, (20,))
    data_loader = DataLoader(TensorDataset(data, labels), batch_size=10)

    # 2. Create a fake model with fake gradient
    pytorch_model = MockModel()

    # 3. Instantiate the PGD attack
    attack = _make_attack()

    # 4. Create the adversarial trainer and run it with a raw Module
    optimizer = SGD(pytorch_model.parameters(), lr=0.01)
    criterion = MockLoss()
    trainer = AdversarialTrainer(optimizer=optimizer, loss=criterion, epochs=1)
    trained_model = trainer.train(pytorch_model, data_loader, attack)
    assert isinstance(trained_model, torch.nn.Module)


def test_adversarial_trainer_with_classifier() -> None:
    # 1. Create random dataset
    data = torch.randn(20, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 9, (20,))
    data_loader = DataLoader(TensorDataset(data, labels), batch_size=10)

    # 2. Create a fake model wrapped in BasePyTorchClassifier
    pytorch_model = MockModel()
    classifier = BasePyTorchClassifier(pytorch_model)

    # 3. Instantiate the PGD attack
    attack = _make_attack()

    # 4. Create the adversarial trainer and run it with a BasePyTorchClassifier
    optimizer = SGD(pytorch_model.parameters(), lr=0.01)
    criterion = MockLoss()
    trainer = AdversarialTrainer(optimizer=optimizer, loss=criterion, epochs=1)
    trained_model = trainer.train(classifier, data_loader, attack)
    assert isinstance(trained_model, torch.nn.Module)


def test_adversarial_trainer_mix_mode_not_implemented() -> None:
    # 1. Create random dataset
    data = torch.randn(20, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 9, (20,))
    data_loader = DataLoader(TensorDataset(data, labels), batch_size=10)

    # 2. Create a fake model with fake gradient
    pytorch_model = MockModel()

    # 3. Instantiate the PGD attack
    attack = _make_attack()

    # 4. Run the trainer with the "mix" combining mode and expect it to fail
    optimizer = SGD(pytorch_model.parameters(), lr=0.01)
    criterion = MockLoss()
    trainer = AdversarialTrainer(optimizer=optimizer, loss=criterion, epochs=1)
    with pytest.raises(NotImplementedError):
        trainer.train(pytorch_model, data_loader, attack, combining_mode="mix")
