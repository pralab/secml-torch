from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from robustbench.utils import download_gdrive
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD, PGDNative
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.optimization.losses import LogitDifferenceLoss
from secmlt.trackers import (
    GradientNormTracker,
    LossTracker,
)
from torch import nn
from torch.utils.data import DataLoader, Subset


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 200)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = torch.relu(self.linear1(x))
        x = self.dropout(x)

        x = torch.relu(self.linear2(x))
        return self.linear3(x)


MODEL_ID = "1s7Kfa2Bs5nY2zLd6dVAxUqNbCNQhPYxs"

model = MNISTModel()
path = Path("models/mnist_distilled.pt")
if not path.exists():
    download_gdrive(MODEL_ID, path)
state_dict = torch.load(
    path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
model.load_state_dict(state_dict)
model.eval()


test_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(),
    train=False,
    root="data",
    download=True,
)
test_dataset = Subset(test_dataset, list(range(3)))
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


device = "cpu"

# Wrap model
secmlt_model = BasePytorchClassifier(model)

# Test accuracy on original data
accuracy = Accuracy()(secmlt_model, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")

# Create and run attack
epsilon = 0.3
num_steps = 200
step_size = 0.005
perturbation_model = LpPerturbationModels.LINF
y_target = None

trackers = [
    LossTracker(),
    GradientNormTracker(),
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
native_adv_ds = native_attack(secmlt_model, test_loader)

robust_accuracy = accuracy = Accuracy()(secmlt_model, native_adv_ds)
print(f"robust accuracy: {robust_accuracy.item():.2f}")


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

loss_tracker = trackers[0]
axes[0].plot(loss_tracker.get().T)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss during PGD attack")

gradient_norm_tracker = trackers[1]
axes[1].plot(gradient_norm_tracker.get().T)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Gradient Norm")
axes[1].set_title("Gradient Norm during PGD attack")

plt.show()


class PGDWithDifferenceOfLogits(PGDNative):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = LogitDifferenceLoss()


trackers = [
    LossTracker(),
    GradientNormTracker(),
]

adaptive_attack = PGDWithDifferenceOfLogits(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
    trackers=trackers,
)

adaptive_attack = adaptive_attack(secmlt_model, test_loader)

robust_accuracy = accuracy = Accuracy()(secmlt_model, adaptive_attack)
print(f"robust accuracy: {robust_accuracy.item():.2f}")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

print(loss_tracker.get().shape)
loss_tracker = trackers[0]
axes[0].plot(loss_tracker.get().T)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss during PGD attack")

gradient_norm_tracker = trackers[1]
axes[1].plot(gradient_norm_tracker.get().T)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Gradient Norm")
axes[1].set_title("Gradient Norm during PGD attack")

plt.show()
