import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor

HF_DATASETS = {
    "mnist": ("ylecun/mnist", "image"),
    "cifar10": ("uoft-cs/cifar10", "img"),
}


class HuggingFaceDataset(Dataset):
    """Wrap a Hugging Face image classification dataset as a PyTorch dataset."""

    def __init__(self, name, train=True, path=None, transform=None):
        repo_id, self.image_key = HF_DATASETS[name]
        split = "train" if train else "test"
        self.dataset = load_dataset(repo_id, split=split, cache_dir=path)
        self.transform = transform if transform is not None else ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample[self.image_key]), torch.tensor(sample["label"])


def get_mnist_dataset(path=None, train=True):
    return HuggingFaceDataset("mnist", train=train, path=path)


def get_cifar10_dataset(path=None, train=True):
    return HuggingFaceDataset("cifar10", train=train, path=path)


def get_mnist_loader(path):
    test_dataset = get_mnist_dataset(path, train=False)
    test_dataset = Subset(test_dataset, list(range(10)))
    return DataLoader(test_dataset, batch_size=10, shuffle=False)
