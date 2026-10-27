from pathlib import Path

import torch
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from loaders.get_loaders import get_mnist_dataset

dataset_path = "example_data/datasets/"
device = "cpu"
net = torch.hub.load("maurapintor/mnist_examples", "mnist_model")
net.to(device)
optimizer = Adam(lr=1e-3, params=net.parameters())
training_dataset = get_mnist_dataset(dataset_path, train=True)
training_data_loader = DataLoader(training_dataset, batch_size=64, shuffle=False)
test_dataset = get_mnist_dataset(dataset_path, train=False)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training MNIST model
trainer = BasePyTorchTrainer(optimizer, epochs=1)
trainer.train(net, training_data_loader)

# Test MNIST model
accuracy = Accuracy()(net, test_data_loader)
print("test accuracy: ", accuracy)

model_path = Path("example_data/models/mnist")
torch.save(net.state_dict(), model_path / "mnist_model.pt")
