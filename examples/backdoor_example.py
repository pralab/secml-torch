import torch
import torchvision.datasets
from models.mnist_net import MNISTNet
from secmlt.adv.backdoor.base_pytorch_backdoor import BackdoorDatasetPyTorch
from secmlt.metrics.classification import Accuracy, AttackSuccessRate
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader


class MNISTBackdoor(BackdoorDatasetPyTorch):
    def _add_trigger(self, x: torch.Tensor) -> torch.Tensor:
        x[:, 0, 24:28, 24:28] = 1.0
        return x


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
target_label = 1
backdoored_mnist = MNISTBackdoor(
    training_dataset, trigger_label=target_label, portion=0.1
)

training_data_loader = DataLoader(backdoored_mnist, batch_size=20, shuffle=False)
test_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(),
    train=False,
    root=dataset_path,
    download=True,
)
test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

trainer = BasePyTorchTrainer(optimizer, epochs=5)
model = BasePytorchClassifier(net, trainer=trainer)
model.train(training_data_loader)

# test accuracy without backdoor
accuracy = Accuracy()(model, test_data_loader)
print("test accuracy: ", accuracy)

# test accuracy on backdoored dataset
backdoored_test_set = MNISTBackdoor(test_dataset)
backdoored_loader = DataLoader(backdoored_test_set, batch_size=20, shuffle=False)

asr = AttackSuccessRate(y_target=target_label)(model, backdoored_loader)
print(f"asr: {asr}")
