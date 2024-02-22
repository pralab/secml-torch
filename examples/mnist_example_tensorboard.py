import os
from secml2.trackers.image_trackers import GradientsTracker, SampleTracker
from secml2.trackers.trackers import (
    GradientNormTracker,
    LossTracker,
    PredictionTracker,
    PerturbationNormTracker,
    TensorboardTracker,
)
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from robustbench.utils import download_gdrive
from secml2.adv.backends import Backends
from secml2.adv.evasion.pgd import PGD
from secml2.adv.evasion.perturbation_models import PerturbationModels

from secml2.metrics.classification import Accuracy
from secml2.models.pytorch.base_pytorch_nn import BasePytorchClassifier


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
epsilon = 0.2
num_steps = 200
step_size = 0.01
perturbation_model = PerturbationModels.LINF
y_target = None

trackers = [
    LossTracker(),
    PredictionTracker(),
    PerturbationNormTracker("linf"),
    GradientNormTracker(),
    SampleTracker(),
    GradientsTracker(),
]

tensorboard_tracker = TensorboardTracker("logs/pgd", trackers)

native_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
    trackers=tensorboard_tracker,
)
native_adv_ds = native_attack(model, test_data_loader)
