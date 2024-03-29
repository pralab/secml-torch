import torch
from loaders.get_loaders import get_mnist_loader
from models.mnist_net import get_mnist_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.aggregators.ensemble import FixedEpsilonEnsemble
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import (
    Accuracy,
    AccuracyEnsemble,
    AttackSuccessRate,
    EnsembleSuccessRate,
)
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

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
epsilon = 0.15
num_steps = 3
step_size = 0.05
perturbation_model = LpPerturbationModels.LINF
y_target = None

pgd_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=True,
    y_target=y_target,
    backend=Backends.NATIVE,
)

multiple_attack_results = [pgd_attack(model, test_loader) for i in range(3)]
criterion = FixedEpsilonEnsemble(loss_fn=torch.nn.CrossEntropyLoss())
best_advs = criterion(model, test_loader, multiple_attack_results)

# Test accuracy on best adversarial examples
n_robust_accuracy = Accuracy()(model, best_advs)
print(f"RA best advs: {n_robust_accuracy.item():.2f}")

# Test accuracy on ensemble
n_robust_accuracy = AccuracyEnsemble()(model, multiple_attack_results)
print(f"RA ensemble: {n_robust_accuracy.item():.2f}")

n_asr = EnsembleSuccessRate(y_target=y_target)(model, multiple_attack_results)
print(f"ASR ensemble: {n_asr.item():.2f}")

for i, res in enumerate(multiple_attack_results):
    n_robust_accuracy = Accuracy()(model, res)
    print(f"RA attack: {i}: {n_robust_accuracy.item():.2f}")

    asr = AttackSuccessRate(y_target=y_target)(model, res)
    print(f"ASR attack: {i}: {asr.item():.2f}")
