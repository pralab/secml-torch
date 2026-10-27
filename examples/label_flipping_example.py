import torch
from secmlt.adv.poisoning.base_data_poisoning import PoisoningDatasetPyTorch
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from loaders.get_loaders import get_mnist_dataset


def flip_label(label):
    return 0 if label != 0 else 1


dataset_path = "example_data/datasets/"
device = "cpu"
net = torch.hub.load("maurapintor/mnist_examples", "mnist_model")
net.to(device)

optimizer = Adam(lr=1e-3, params=net.parameters())
training_dataset = get_mnist_dataset(dataset_path, train=True)
target_label = 1
poisoned_mnist = PoisoningDatasetPyTorch(
    training_dataset,
    label_manipulation_func=flip_label,
    portion=0.4,
)

training_data_loader = DataLoader(training_dataset, batch_size=20, shuffle=False)
poisoned_data_loader = DataLoader(poisoned_mnist, batch_size=20, shuffle=False)

test_dataset = get_mnist_dataset(dataset_path, train=False)
test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

for k, data_loader in {
    "normal": training_data_loader,
    "poisoned": poisoned_data_loader,
}.items():
    trainer = BasePyTorchTrainer(optimizer, epochs=3)
    trainer.train(net, data_loader)
    # test accuracy without backdoor
    accuracy = Accuracy()(net, test_data_loader)
    print(f"test accuracy on {k} data: {accuracy.item():.3f}")
