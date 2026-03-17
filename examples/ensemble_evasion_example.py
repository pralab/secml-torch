import torchvision.datasets
from robustbench.utils import load_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from torch.utils.data import DataLoader, Subset

from secmlt.adv.evasion.ensemble.ensemble_pgd import EnsemblePGD
from secmlt.adv.evasion.ensemble.loss.avg_loss import AvgEnsembleLoss
from secmlt.adv.evasion.modular_attacks.modular_attack import CE_LOSS
from secmlt.models.ensemble.ensemble_model import EnsembleModel
from secmlt.models.ensemble.ensemble_function import (
    AvgEnsembleFunction, # forward averages the outputs of the ensemble models.
    RandomEnsembleFunction, # forward considers one randomly picked model.
    RawEnsembleFunction # forward returns the raw outputs of the models.
)


# We load models to be included in the ensemble
nets = [
    load_model(model_name="Ding2020MMA", dataset="cifar10", threat_model="L2"),
    load_model(model_name="Rony2019Decoupling", dataset="cifar10", threat_model="L2")
]

device = "cpu"

models = []

# Wrap each model
for net in nets:
    net.to(device)
    models.append(BasePytorchClassifier(net))


test_dataset = torchvision.datasets.CIFAR10(
    transform=torchvision.transforms.ToTensor(),
    train=False,
    root=".",
    download=True,
)
test_dataset = Subset(test_dataset, list(range(5)))
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Setup the ensemble parameters
ensemble_fct = RawEnsembleFunction()
loss_function = AvgEnsembleLoss() # CE_LOSS if other ensemble functions are used

# Wrap the models in the ensemble using the defined ensemble function
model = EnsembleModel(models={f"model_{i}": net for i, net \
    in enumerate(models)}, ensemble_function=ensemble_fct)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print("Accuracy:", accuracy.item())

# Create and run attack
epsilon = 0.5
num_steps = 10
step_size = 0.005
perturbation_model = LpPerturbationModels.LINF
y_target = None
native_attack = EnsemblePGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    loss_function=loss_function,
    y_target=y_target,
    trackers=None,
)


native_adv_ds = native_attack(model, test_data_loader)

# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print("Robust Accuracy (PGD Native): ", n_robust_accuracy.item())
