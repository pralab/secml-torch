import os
from secmlt.trackers.trackers import (
    LossTracker,
    PredictionTracker,
    PerturbationNormTracker,
)
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from robustbench.utils import download_gdrive
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.pgd import PGD
from secmlt.adv.evasion.perturbation_models import PerturbationModels

from secmlt.metrics.classification import Accuracy, SampleWiseAccuracy
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
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print("Accuracy on clean samples: ", accuracy)

# Create and run attack
epsilon = 0.05
num_steps = 10
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None

trackers = [
    LossTracker(),
    PredictionTracker(),
    PerturbationNormTracker("linf"),
]

attack_1 = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
)

attack_2 = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=True,
    y_target=y_target,
    backend=Backends.FOOLBOX,
)

attack_3 = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=0.01,
    random_start=True,
    y_target=y_target,
    backend=Backends.FOOLBOX,
)

# run all attacks and collect results
adv_datasets = []
for i, attack in enumerate([attack_1, attack_2, attack_3]):
    adv_dataset = attack(model, test_data_loader)
    adv_datasets.append(adv_dataset)
    # individual attacks robust accuracy
    robust_accuracy = Accuracy()(model, adv_dataset)
    print(f"robust accuracy attack #{i}: {robust_accuracy}")

# test accuracy on ensemble
n_robust_accuracy = SampleWiseAccuracy()(model, adv_datasets)
print("robust accuracy: ", n_robust_accuracy)
