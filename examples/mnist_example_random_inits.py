import os
from secmlt.adv.evasion.aggregators.ensemble import FixedEpsilonEnsemble
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from robustbench.utils import download_gdrive
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.pgd import PGD
from secmlt.adv.evasion.perturbation_models import PerturbationModels
from secmlt.metrics.classification import (
    Accuracy,
    AccuracyEnsemble,
    AttackSuccessRate,
    EnsembleSuccessRate,
)
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


device = "cpu"
net = MNISTNet()
model_folder = "models/mnist"
model_weights_path = os.path.join("mnist_model.pt")
if not os.path.exists(model_weights_path):
    os.makedirs(model_folder, exist_ok=True)
    MODEL_ID = "12h1tXK442jHSE7wtsPpt8tU8f04R4nHM"
    download_gdrive(MODEL_ID, model_weights_path)

model_weigths = torch.load(model_weights_path, map_location=device)
net.eval()
net.load_state_dict(model_weigths)
test_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(), train=False, root=".", download=True
)
test_dataset = Subset(test_dataset, list(range(10)))
test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print(f"test accuracy: {accuracy.item():.2f}")

# Create and run attack
epsilon = 0.15
num_steps = 3
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None

pgd_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=True,
    y_target=y_target,
    backend=Backends.NATIVE,
)

multiple_attack_results = [pgd_attack(model, test_data_loader) for i in range(3)]
criterion = FixedEpsilonEnsemble(loss_fn=torch.nn.CrossEntropyLoss())
best_advs = criterion(model, test_data_loader, multiple_attack_results)

# Test accuracy on best adversarial examples
n_robust_accuracy = Accuracy()(model, best_advs)
print(f"RA best advs: {n_robust_accuracy.item():.2f}")

# Test accuracy on ensemble
n_robust_accuracy = AccuracyEnsemble()(model, multiple_attack_results)
print(f"RA ensemble: {n_robust_accuracy.item():.2f}")

n_asr = EnsembleSuccessRate(y_target=y_target)(model, multiple_attack_results)
print(f"ASR ensemble: {n_asr.item():.2f}")

for i, res in enumerate(multiple_attack_results):
    n_robust_accuracy = Accuracy()(model, res)
    print(f"RA attack: {i}: {n_robust_accuracy.item():.2f}")

    asr = AttackSuccessRate(y_target=y_target)(model, res)
    print(f"ASR attack: {i}: {asr.item():.2f}")
