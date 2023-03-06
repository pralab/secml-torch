import torch
import torchvision.datasets
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset, TensorDataset
from src.adv.backends import Backends
from src.adv.evasion.pgd import PGD
from src.adv.evasion.perturbation_models import PerturbationModels

from src.metrics.classification import Accuracy
from src.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from robustbench.utils import load_model


def attack_linf_pytorch_optim(model, samples, labels, optimizer, steps=100,
                              step_size=0.05, eps=0.3, device='cpu'):
    x_adv = samples.clone().detach().to(device).requires_grad_()
    optimizer = optimizer([x_adv], lr=step_size)
    for _ in range(steps):
        out = model(x_adv)
        loss = -torch.nn.functional.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        x_adv.grad = x_adv.grad.sign()
        optimizer.step()

        diff = x_adv.data - samples
        diff = diff.clamp_(-eps, +eps)
        x_adv.detach().copy_((diff + samples).clamp_(0, 1))
    return x_adv


net = load_model(model_name="Rony2019Decoupling", dataset="cifar10", threat_model="L2")
device = "mps"
net.to(device)
test_dataset = torchvision.datasets.CIFAR10(
    transform=torchvision.transforms.ToTensor(), train=False, root=".", download=True
)
test_dataset = Subset(test_dataset, list(range(1)))
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print(accuracy)

# Create and run attack
epsilon = 100
num_steps = 50
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None
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
print(n_robust_accuracy)

print('OTHER ATTACK')

advs = []
advlb = []
for data, labels in test_data_loader:
    x_adv = attack_linf_pytorch_optim(model, data.to(device), labels.to(device), SGD, num_steps, step_size, epsilon,
                                      'mps')
    advs.append(x_adv)
    advlb.append(labels)
advlb = torch.cat(advlb)
advs = torch.cat(advs)
f_adv_ds = DataLoader(TensorDataset(advs, advlb), batch_size=5, shuffle=False)
# Test accuracy on adversarial examples
f_robust_accuracy = Accuracy()(model, f_adv_ds)
print(f_robust_accuracy)

native_data, native_labels = next(iter(native_adv_ds))
f_data, f_labels = next(iter(f_adv_ds))

distance = torch.linalg.norm(native_data.flatten(start_dim=1).to(device) - f_data.flatten(start_dim=1),
                             ord=float('inf'), dim=1)
print(distance)
