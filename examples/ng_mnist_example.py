from loaders.get_loaders import get_mnist_loader
from models.mnist_net import get_mnist_model
from secmlt.adv.evasion.nevergrad_optim.ng_attacks import NevergradGeneticAlgorithm
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.metrics.classification import Accuracy
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
epsilon = 1
num_steps = 100
perturbation_model = LpPerturbationModels.LINF
y_target = None

native_attack = NevergradGeneticAlgorithm(
    perturbation_model=perturbation_model,
    population_size=30,
    epsilon=epsilon,
    num_steps=num_steps,
    budget=num_steps,
    random_start=True,
    y_target=y_target,
)

native_adv_ds = native_attack(model, test_loader)

# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print("robust accuracy native: ", n_robust_accuracy)
