import torch
import torchvision.datasets
from torch.optim import Adam
from torch.utils.data import DataLoader

from secml2.metrics.classification import Accuracy
from secml2.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secml2.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer


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
net.to(device)
optimizer = Adam(lr=1e-3, params=net.parameters())
training_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(), train=True, root=".", download=True
)
training_data_loader = DataLoader(training_dataset, batch_size=64, shuffle=False)
test_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(), train=False, root=".", download=True
)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training MNIST model
trainer = BasePyTorchTrainer(optimizer, epochs=1)
model = BasePytorchClassifier(net, trainer=trainer)
model.train(training_data_loader)

# Test MNIST model
accuracy = Accuracy()(model, test_data_loader)
print(accuracy)

torch.save(model.model.state_dict(), "mnist_model.pt")
