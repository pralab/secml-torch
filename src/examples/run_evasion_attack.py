import torch
import torchvision.datasets
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from src.adv.backends import Backends
from src.adv.evasion.pgd import PGD
from src.adv.evasion.threat_models import ThreatModels

from src.metrics.classification import Accuracy
from src.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from src.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer

from robustbench.utils import load_model

net = load_model(model_name='Rony2019Decoupling', dataset='cifar10', threat_model='L2')
net.to('cpu')
test_dataset = torchvision.datasets.CIFAR10(transform=torchvision.transforms.ToTensor(), train=False, root='.',
                                          download=True)
test_dataset = Subset(test_dataset, list(range(5)))
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print(accuracy)

# Create and run attack
attack = PGD(threat_model=ThreatModels.LINF, epsilon=0.5, num_steps=50, step_size=0.05, random_start=False, y_target=None, backend=Backends.FOOLBOX)
adv_ds = attack(model, test_data_loader)

# Test accuracy on adversarial examples
robust_accuracy = Accuracy()(model, adv_ds)
print(robust_accuracy)