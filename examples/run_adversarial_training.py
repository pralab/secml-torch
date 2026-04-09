import torch
import torchvision.datasets
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGDNative
from secmlt.defenses.adv_training.pytorch.adversarial_trainer import AdversarialTrainer
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from torch.optim import Adam
from torch.utils.data import DataLoader

dataset_path = "example_data/datasets/"

REPO_LINK = "chenyaofo/pytorch-cifar-models"
DEVICE = "mps"
MEAN = (0.49139968, 0.48215841, 0.44653091)
STD = (0.2023, 0.1994, 0.2010)


# define model
model = torch.hub.load(REPO_LINK, "cifar10_resnet20", pretrained=True)

bn = torch.nn.BatchNorm2d(3, affine=False)
bn.running_mean = torch.tensor(MEAN)
bn.running_var = torch.tensor([std**2 for std in STD])
model = torch.nn.Sequential(bn, model)
model.to(DEVICE)


optimizer = Adam(lr=1e-3, params=model.parameters())
training_dataset = torchvision.datasets.CIFAR10(
    transform=torchvision.transforms.ToTensor(),
    train=True,
    root=dataset_path,
    download=True,
)
training_dataset = torch.utils.data.Subset(training_dataset, range(1000))
training_data_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(
    transform=torchvision.transforms.ToTensor(),
    train=False,
    root=dataset_path,
    download=True,
)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inizialize the PGD attack
attack_train = PGDNative(
    perturbation_model=LpPerturbationModels.LINF,
    epsilon=0.05,
    num_steps=3,
    step_size=0.01,
    random_start=False,
    y_target=None,
)

attack_eval = PGDNative(
    perturbation_model=LpPerturbationModels.LINF,
    epsilon=0.01,
    num_steps=3,
    step_size=0.01,
    random_start=False,
    y_target=None,
)

# Evauate the model on the test set before training
accuracy = Accuracy()(BasePyTorchClassifier(model), test_data_loader)
print("Accuracy before training: ", accuracy)
# Evaluate the model on the test set with adversarial examples before training
adv_loader = attack_eval(BasePyTorchClassifier(model), test_data_loader)
adv_accuracy = Accuracy()(BasePyTorchClassifier(model), adv_loader)
print("Robust Accuracy before training: ", adv_accuracy)

# Training CIFAR10 model
trainer = AdversarialTrainer(optimizer, epochs=10)
trainer.train(model, training_data_loader, attack_train)

# Evaluate the model on the test set after training
accuracy = Accuracy()(BasePyTorchClassifier(model), test_data_loader)
print("Accuracy after training: ", accuracy)
# Evaluate the model on the test set with adversarial examples after training
adv_loader = attack_eval(BasePyTorchClassifier(model), test_data_loader)
adv_accuracy = Accuracy()(BasePyTorchClassifier(model), adv_loader)
print("Robust Accuracy after training: ", adv_accuracy)
