"""Example: tracking a Foolbox attack using ModelTracker.

Trackers can be passed directly to the Foolbox/AdvLib attack — the model
is wrapped internally with ModelTracker, so no manual wrapping is needed.
A raw nn.Module can also be passed directly to the attack.
"""

import importlib.util

import torch
from loaders.get_loaders import get_mnist_loader
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy
from secmlt.trackers.trackers import (
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
)

device = "mps"
dataset_path = "example_data/datasets/"
net = torch.hub.load("maurapintor/distilled_mnist", "mnist_model", weights="teacher")
net.eval()
test_loader = get_mnist_loader(dataset_path)

# Test accuracy on original data
accuracy = Accuracy()(net, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")

epsilon = 1
num_steps = 10
step_size = 0.05
perturbation_model = LpPerturbationModels.LINF

available_backends = [Backends.NATIVE]
if importlib.util.find_spec("foolbox") is not None:
    available_backends.append(Backends.FOOLBOX)
if importlib.util.find_spec("adv_lib") is not None:
    available_backends.append(Backends.ADVLIB)

print("\nRunning PGD with model-level tracking on available backends...")
for backend in available_backends:
    trackers = [
        LossTracker(),
        PredictionTracker(),
        PerturbationNormTracker(LpPerturbationModels.LINF),
    ]

    attack = PGD(
        perturbation_model=perturbation_model,
        epsilon=epsilon,
        num_steps=num_steps,
        step_size=step_size,
        random_start=False,
        y_target=None,
        backend=backend,
        trackers=trackers,
    )

    # Pass raw nn.Module: wrappers internally use ModelTracker where needed.
    adv_ds = attack(net, test_loader)
    robust_accuracy = Accuracy()(net, adv_ds)
    print(f"\n[{backend}] robust accuracy: {robust_accuracy.item():.4f}")

    for tracker in trackers:
        tracked = tracker.get()
        print(f"[{backend}] {tracker.name} shape: {tuple(tracked.shape)}")
