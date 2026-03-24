import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGDNative
from secmlt.defenses.adv_training.pytorch.adversarial_trainer import AdversarialTrainer
from secmlt.tests.mocks import MockLoss, MockModel
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset


def test_adversarial_trainer() -> None:
    # 1. Create random dataset
    data = torch.randn(20, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 9, (20,))
    data_loader = DataLoader(TensorDataset(data, labels), batch_size=10)

    # 2. Create a fake model with fake gradient
    pytorch_model = MockModel()

    # 4. Instantiate the PGD attack
    attack = PGDNative(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.3,
        num_steps=3,
        step_size=0.1,
        random_start=False,
        y_target=None,
    )

    # 5. Create the adversarial trainer and run it
    optimizer = SGD(pytorch_model.parameters(), lr=0.01)
    criterion = MockLoss()
    trainer = AdversarialTrainer(optimizer=optimizer, loss=criterion, epochs=1)
    trained_model = trainer.train(pytorch_model, data_loader, attack)
    assert isinstance(trained_model, torch.nn.Module)
