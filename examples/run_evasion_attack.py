import torchvision.datasets
from robustbench.utils import load_model
from torch.utils.data import DataLoader, Subset

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

net = load_model(model_name="Rony2019Decoupling", dataset="cifar10", threat_model="L2")
device = "cpu"
net.to(device)
test_dataset = torchvision.datasets.CIFAR10(
    transform=torchvision.transforms.ToTensor(),
    train=False,
    root=".",
    download=True,
)
test_dataset = Subset(test_dataset, list(range(5)))
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print("Accuracy:", accuracy.item())

# Create and run attack
epsilon = 0.5
num_steps = 10
step_size = 0.005
perturbation_model = LpPerturbationModels.LINF
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
print("Robust Accuracy (PGD Native): ", n_robust_accuracy.item())

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
print("Robust Accuracy (PGD Foolbox): ", n_robust_accuracy.item())

# Create and run attack
advlib_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    loss_function="dlr",
    y_target=y_target,
    backend=Backends.ADVLIB,
)
al_adv_ds = advlib_attack(model, test_data_loader)

# Test accuracy on adversarial examples
f_robust_accuracy = Accuracy()(model, al_adv_ds)
print("Robust Accuracy (PGD AdvLib): ", n_robust_accuracy.item())
