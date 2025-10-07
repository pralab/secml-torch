from loaders.get_loaders import get_mnist_loader
from models.mnist_net import get_mnist_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.ddn import DDN
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

# Attack configuration
num_steps = 100
eps_init = 1.0
gamma = 0.05
perturbation_model = LpPerturbationModels.L2
y_target = None

# Native backend
native_attack = DDN(
    num_steps=num_steps,
    eps_init=eps_init,
    gamma=gamma,
    y_target=y_target,
    backend=Backends.NATIVE,
)
native_adv_ds = native_attack(model, test_loader)

# Foolbox backend
foolbox_attack = DDN(
    num_steps=num_steps,
    eps_init=eps_init,
    gamma=gamma,
    y_target=y_target,
    backend=Backends.FOOLBOX,
)
f_adv_ds = foolbox_attack(model, test_loader)

# Adversarial Library backend
advlib_attack = DDN(
    num_steps=num_steps,
    eps_init=eps_init,
    gamma=gamma,
    y_target=y_target,
    backend=Backends.ADVLIB,
)
al_adv_ds = advlib_attack(model, test_loader)

# Evaluate robustness
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print("robust accuracy native: ", n_robust_accuracy)

f_robust_accuracy = Accuracy()(model, f_adv_ds)
print("robust accuracy foolbox: ", f_robust_accuracy)

al_robust_accuracy = Accuracy()(model, al_adv_ds)
print("robust accuracy AdvLib: ", al_robust_accuracy)

# Inspect perturbation norms
native_data, _ = next(iter(native_adv_ds))
f_data, _ = next(iter(f_adv_ds))
adv_lib_data, _ = next(iter(al_adv_ds))
real_data, _ = next(iter(test_loader))

p = LpPerturbationModels.get_p(perturbation_model)
distances_native = (real_data - native_data).flatten(start_dim=1).norm(p=p, dim=-1)
distances_foolbox = (real_data - f_data).flatten(start_dim=1).norm(p=p, dim=-1)
distances_advlib = (real_data - adv_lib_data).flatten(start_dim=1).norm(p=p, dim=-1)
print("Native distances: ", distances_native)
print("Foolbox distances: ", distances_foolbox)
print("AdvLib distances: ", distances_advlib)
