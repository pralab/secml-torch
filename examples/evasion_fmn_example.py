import importlib.util

import torch
from loaders.get_loaders import get_mnist_loader
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.fmn import FMN
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.metrics.classification import Accuracy
from secmlt.optimization.losses import LogitDifferenceLoss
from secmlt.trackers.trackers import (
    GradientNormTracker,
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
)

HAS_FOOLBOX = importlib.util.find_spec("foolbox") is not None
HAS_ADVLIB = importlib.util.find_spec("adv_lib") is not None

device = "cpu"
dataset_path = "example_data/datasets/"
net = torch.hub.load("maurapintor/distilled_mnist", "mnist_model", weights="teacher")
net.eval()
test_loader = get_mnist_loader(dataset_path)

# Test accuracy on original data
accuracy = Accuracy()(net, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")

# Create and run attack
num_steps = 200
step_size = 0.05
perturbation_model = LpPerturbationModels.LINF
y_target = None

trackers = [
    LossTracker(loss_fn=LogitDifferenceLoss()),
    PredictionTracker(),
    PerturbationNormTracker(perturbation_model),
    GradientNormTracker(),
]

native_attack = FMN(
    perturbation_model=perturbation_model,
    num_steps=num_steps,
    step_size=step_size,
    y_target=y_target,
    backend=Backends.NATIVE,
    trackers=trackers,
)

native_adv_ds = native_attack(net, test_loader)


# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(net, native_adv_ds)
print("robust accuracy native: ", n_robust_accuracy)

# Trackers are now also supported for Foolbox and AdvLib backends
if HAS_FOOLBOX:
    foolbox_trackers = [
        LossTracker(loss_fn=LogitDifferenceLoss()),
        PredictionTracker(),
        PerturbationNormTracker(perturbation_model),
        GradientNormTracker(),
    ]
    foolbox_attack = FMN(
        perturbation_model=perturbation_model,
        num_steps=num_steps,
        step_size=step_size,
        y_target=y_target,
        backend=Backends.FOOLBOX,
        trackers=foolbox_trackers,
    )
    f_adv_ds = foolbox_attack(net, test_loader)

    # Test accuracy on foolbox
    f_robust_accuracy = Accuracy()(net, f_adv_ds)
    print("robust accuracy foolbox: ", f_robust_accuracy)

if HAS_ADVLIB:
    advlib_trackers = [
        LossTracker(loss_fn=LogitDifferenceLoss()),
        PredictionTracker(),
        PerturbationNormTracker(perturbation_model),
        GradientNormTracker(),
    ]
    advlib_attack = FMN(
        perturbation_model=perturbation_model,
        num_steps=num_steps,
        step_size=step_size,
        y_target=y_target,
        backend=Backends.ADVLIB,
        trackers=advlib_trackers,
    )
    al_adv_ds = advlib_attack(net, test_loader)

    # Test accuracy on adv lib
    al_robust_accuracy = Accuracy()(net, al_adv_ds)
    print("robust accuracy AdvLib: ", al_robust_accuracy)

