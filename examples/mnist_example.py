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

from secmlt.metrics.classification import Accuracy
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
test_dataset = Subset(test_dataset, list(range(5)))
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print(accuracy)

# Create and run attack
epsilon = 0.3
num_steps = 10
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None

trackers = [
    LossTracker(),
    PredictionTracker(),
    PerturbationNormTracker("linf"),
]

native_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
    trackers=trackers,
)
native_adv_ds = native_attack(model, test_data_loader)

for tracker in trackers:
    print(tracker.name)
    print(tracker.get())

# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print("robust accuracy foolbox: ", n_robust_accuracy)

# Create and run attack
foolbox_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.FOOLBOX,
)
f_adv_ds = foolbox_attack(model, test_data_loader)

# Test accuracy on adversarial examples
f_robust_accuracy = Accuracy()(model, f_adv_ds)
print("robust accuracy native: ", f_robust_accuracy)

native_data, native_labels = next(iter(native_adv_ds))
f_data, f_labels = next(iter(f_adv_ds))
real_data, real_labels = next(iter(test_data_loader))

distance = torch.linalg.norm(
    native_data.detach().cpu().flatten(start_dim=1)
    - f_data.detach().cpu().flatten(start_dim=1),
    ord=float("inf"),
    dim=1,
)
print("Solutions are :", distance, "linf distant")
