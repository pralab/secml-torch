import torchvision
from torch.utils.data import DataLoader, Subset


def get_mnist_loader(path):
    test_dataset = torchvision.datasets.MNIST(
        transform=torchvision.transforms.ToTensor(),
        train=False,
        root=path,
        download=True,
    )
    test_dataset = Subset(test_dataset, list(range(10)))
    return DataLoader(test_dataset, batch_size=10, shuffle=False)
