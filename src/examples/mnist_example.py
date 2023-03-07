from collections import OrderedDict
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from src.adv.backends import Backends
from src.adv.evasion.pgd import PGD
from src.adv.evasion.perturbation_models import PerturbationModels

from src.metrics.classification import Accuracy
from src.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from robustbench.utils import load_model
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

net = SmallCNN()
model_weigths = torch.load("models/mnist/mnist_smallcnn_standard.pth")
net.eval()
net.load_state_dict(model_weigths)
device = "cpu"
net.to(device)
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
epsilon = 0.3
num_steps = 1
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None # torch.tensor([9]).to(device)
native_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
)
native_adv_ds = native_attack(model, test_data_loader)

# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print("robust accuracy foolbox: ", n_robust_accuracy)

# Create and run attack
foolbox_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.FOOLBOX,
)
f_adv_ds = foolbox_attack(model, test_data_loader)

# Test accuracy on adversarial examples
f_robust_accuracy = Accuracy()(model, f_adv_ds)
print("robust accuracy native: ", f_robust_accuracy)

native_data, native_labels = next(iter(native_adv_ds))
f_data, f_labels = next(iter(f_adv_ds))
real_data, real_labels = next(iter(test_data_loader))

distance = torch.linalg.norm(native_data.flatten(start_dim=1).to(device) - f_data.flatten(start_dim=1),
                             ord=float('inf'), dim=1)
print("Solutions are :", distance, "linf distant")

# real_native = torch.linalg.norm(torch.flatten(real_data.to(device) - native_data.to(device), start_dim=1), ord=float('inf'), dim=1)
# real_fb = torch.linalg.norm(torch.flatten(real_data.to(device) - f_data.to(device), start_dim=1), ord=float('inf'), dim=1)


# print(real_native, real_fb)
