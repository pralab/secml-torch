import torch
from secmlt.adv.poisoning.backdoor import BackdoorDatasetPyTorch
from secmlt.metrics.classification import Accuracy, AttackSuccessRate
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from loaders.get_loaders import get_mnist_dataset


def apply_patch(x: torch.Tensor) -> torch.Tensor:
    x[:, 0, 24:28, 24:28] = 1.0
    return x


dataset_path = "example_data/datasets/"
device = "cpu"
net = torch.hub.load("maurapintor/mnist_examples", "mnist_model")
net.to(device)
optimizer = Adam(lr=1e-3, params=net.parameters())
training_dataset = get_mnist_dataset(dataset_path, train=True)
target_label = 1
backdoored_mnist = BackdoorDatasetPyTorch(
    training_dataset,
    data_manipulation_func=apply_patch,
    trigger_label=target_label,
    portion=0.1,
)

training_data_loader = DataLoader(backdoored_mnist, batch_size=20, shuffle=False)
test_dataset = get_mnist_dataset(dataset_path, train=False)
test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

trainer = BasePyTorchTrainer(optimizer, epochs=5)
trainer.train(net, training_data_loader)

# test accuracy without backdoor
accuracy = Accuracy()(net, test_data_loader)
print("test accuracy: ", accuracy)

# test accuracy on backdoored dataset
backdoored_test_set = BackdoorDatasetPyTorch(
    test_dataset, data_manipulation_func=apply_patch
)
backdoored_loader = DataLoader(backdoored_test_set, batch_size=20, shuffle=False)

asr = AttackSuccessRate(y_target=target_label)(net, backdoored_loader)
print(f"asr: {asr}")
