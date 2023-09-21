import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from src.adv.backends import Backends
from src.adv.evasion.pgd import PGD
from src.adv.evasion.perturbation_models import PerturbationModels
from src.metrics.classification import Accuracy
from src.models.sklearn.svm import SVM


model = SVM()
device = "cpu"
train_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(), train=True, root=".", download=True
)
train_dataset = Subset(train_dataset, list(range(0, 1000)))
test_dataset = torchvision.datasets.MNIST(
    transform=torchvision.transforms.ToTensor(), train=False, root=".", download=True
)
test_dataset = Subset(test_dataset, list(range(0, 1000)))

train_data_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# train model
model.train(train_data_loader)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print(accuracy)

# Create and run attack
epsilon = 0.5
num_steps = 50
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None

# Create and run attack
attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=None,
    backend=Backends.FOOLBOX,
)
f_adv_ds = attack(model, test_data_loader)

# Test accuracy on adversarial examples
f_robust_accuracy = Accuracy()(model, f_adv_ds)
print(f_robust_accuracy)
