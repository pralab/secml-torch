from pathlib import Path

import torch
import torchvision.datasets
from models.mnist_net import MNISTNet
from torch.optim import Adam
from torch.utils.data import DataLoader

from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer

dataset_path = "example_data/datasets/"
device = "cpu"
net = MNISTNet()
net.to(device)
optimizer = Adam(lr=1e-3, params=net.parameters())
training_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(),
    train=True,
    root=dataset_path,
    download=True,
)
training_data_loader = DataLoader(training_dataset, batch_size=64, shuffle=False)
test_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(),
    train=False,
    root=dataset_path,
    download=True,
)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training MNIST model
trainer = BasePyTorchTrainer(optimizer, epochs=1)
model = BasePytorchClassifier(net, trainer=trainer)
model.train(training_data_loader)

# Test MNIST model
accuracy = Accuracy()(model, test_data_loader)
print("test accuracy: ", accuracy)

model_path = Path("example_data/models/mnist")
torch.save(model.model.state_dict(), model_path / "mnist_model.pt")
