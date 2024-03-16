from pathlib import Path

import torch
from robustbench.utils import download_gdrive


class MNISTNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_mnist_model(path):
    net = MNISTNet()
    path = Path(path)
    model_weights_path = path / "mnist_model.pt"
    if not model_weights_path.exists():
        path.mkdir(exist_ok=True, parents=True)
        model_id = "12h1tXK442jHSE7wtsPpt8tU8f04R4nHM"
        download_gdrive(model_id, model_weights_path)

    model_weigths = torch.load(model_weights_path, map_location="cpu")
    net.eval()
    net.load_state_dict(model_weigths)
    return net
