import torch
from foolbox.attacks.boundary_attack import BoundaryAttack
from loaders.get_loaders import get_mnist_loader
from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secmlt.metrics.classification import Accuracy
from secmlt.trackers.trackers import (
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
)


class BoundaryAttackFoolbox(BaseFoolboxEvasionAttack):
    """Wrapper of the Foolbox implementation of the Boundary Attack (black-box)."""

    def __init__(
        self,
        steps: int = 1000,
        epsilon: float = float("inf"),
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Create Boundary Attack with Foolbox backend.

        Parameters
        ----------
        steps : int, optional
            Number of iterations for the attack, by default 1000.
        epsilon : float, optional
            Maximum perturbation allowed. Default is inf (no limit).
        y_target : int | None, optional
            Target label for the attack, None for untargeted, by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        """
        foolbox_attack = BoundaryAttack(steps=steps)

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs,
        )




device = "cpu"
dataset_path = "example_data/datasets/"
net = torch.hub.load("maurapintor/distilled_mnist", "mnist_model", weights="teacher")
net.eval()

# Get a small subset for speed (Boundary Attack is slow)
test_loader = get_mnist_loader(dataset_path)

# Test accuracy on original data
accuracy = Accuracy()(net, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")

# Setup trackers for monitoring the attack
trackers = [
    LossTracker(),
    PredictionTracker(),
    PerturbationNormTracker(),
]

boundary_attack = BoundaryAttackFoolbox(
    steps=30,  # Boundary attack typically needs more iterations
    epsilon=float("inf"),
    y_target=None,  # Untargeted attack
    trackers=trackers,
)

# Run attack
ba_adv_ds = boundary_attack(net, test_loader)

# Test accuracy on adversarial examples
ba_robust_accuracy = Accuracy()(net, ba_adv_ds)
print(f"robust accuracy (Boundary Attack): {ba_robust_accuracy.item():.4f}")
