import torch
from loaders.get_loaders import get_mnist_loader
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy

device = "cpu"
dataset_path = "example_data/datasets/"
net = torch.hub.load("maurapintor/distilled_mnist", "mnist_model", weights="student")
net.eval()
test_loader = get_mnist_loader(dataset_path)

# Test accuracy on original data
accuracy = Accuracy()(net, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")


# Create and run attack
epsilon = 0.3
num_steps = 10
step_size = 0.05
perturbation_model = LpPerturbationModels.LINF
y_target = None

attack_1 = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
)
attack_2 = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
)

attack_2.initializer = attack_1


adv_ds = attack_2(net, test_loader)


# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(net, adv_ds)
print("robust accuracy: ", n_robust_accuracy.item())
