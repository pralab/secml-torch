import torch
from loaders.get_loaders import get_mnist_loader
from models.mnist_net import get_mnist_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.trackers.trackers import (
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
)

device = "cpu"
model_path = "example_data/models/mnist"
dataset_path = "example_data/datasets/"
net = get_mnist_model(model_path).to(device)
test_loader = get_mnist_loader(dataset_path)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")

# Create and run attack
epsilon = 1
num_steps = 10
step_size = 0.05
perturbation_model = LpPerturbationModels.LINF
y_target = None

trackers = [
    LossTracker(),
    PredictionTracker(),
    PerturbationNormTracker(perturbation_model),
]

native_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
    trackers=trackers,
)

native_adv_ds = native_attack(model, test_loader)

for tracker in trackers:
    print(tracker.name)
    print(tracker.get())

# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print("robust accuracy native: ", n_robust_accuracy)

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
f_adv_ds = foolbox_attack(model, test_loader)

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
al_adv_ds = advlib_attack(model, test_loader)

# Test accuracy on foolbox
f_robust_accuracy = Accuracy()(model, f_adv_ds)
print("robust accuracy foolbox: ", f_robust_accuracy)

# Test accuracy on adv lib
al_robust_accuracy = Accuracy()(model, al_adv_ds)
print("robust accuracy AdvLib: ", al_robust_accuracy)

native_data, native_labels = next(iter(native_adv_ds))
f_data, f_labels = next(iter(f_adv_ds))
real_data, real_labels = next(iter(test_loader))


distance = torch.linalg.norm(
    native_data.detach().cpu().flatten(start_dim=1)
    - f_data.detach().cpu().flatten(start_dim=1),
    ord=LpPerturbationModels.pert_models[perturbation_model],
    dim=1,
)
print("Solutions are :", distance, f"{perturbation_model} distant")
