 # SecML2: A Library for Robustness Evaluation of Deep Learning Models

SecML2 is an open-source Python library designed to facilitate research in the area of Adversarial Machine Learning (AML) and robustness evaluation. 
The library provides a simple yet powerful interface for generating various types of adversarial examples, as well as tools for evaluating the robustness of machine learning models against such attacks.

## Installation

You can install SecML2 via pip:
```bash
pip install secml2
```

This will install the core version of SecML2, including only the main functionalities such as native implementation of attacks and PyTorch wrappers.

### Install with extras

The library can be installed together with other plugins that enable further functionalities. 

* [Foolbox](https://github.com/bethgelab/foolbox), a Python toolbox to create adversarial examples.
* [Tensorboard](https://www.tensorflow.org/tensorboard), a visualization toolkit for machine learning experimentation.

Install one or more extras with the command:
```bash
pip install secml2[foolbox,tensorboard]
```

## Key Features

- **Built for Deep Learning:** SecML2 is compatible with the popular machine learning framework PyTorch.
- **Various types of adversarial attacks:** SecML2 includes support for a wide range of attack methods (evasion, poisoning, ...) such as different implementations imported from popular AML libraries (Foolbox, Adversarial Library).
- **Customizable attacks:** SecML2 offers several levels of analysis for the models, including modular implementations of existing attacks to extend with different loss functions, optimizers, and more.
- **Attack debugging:** Built-in debugging of evaluations by logging events and metrics along the attack runs (even on Tensorboard).

## Usage

Here's a brief example of using SecML2 to evaluate the robustness of a trained classifier:

```python
from secml2.adv.evasion.pgd import PGD
from secml2.metrics.classification import Accuracy
from secml2.models.pytorch.base_pytorch_nn import BasePytorchClassifier


model = ...
torch_data_loader = ...

# Wrap model
model = BasePytorchClassifier(model)

# create and run attack
attack = PGD(
    perturbation_model="l2",
    epsilon=0.4,
    num_steps=100,
    step_size=0.01,
)

adversarial_loader = attack(model, torch_data_loader)

# Test accuracy on adversarial examples
robust_accuracy = Accuracy()(model, adversarial_loader)
```

For more detailed usage instructions and examples, please refer to the [official documentation](https://secml2.readthedocs.io/en/latest/) or to the [examples](https://github.com/pralab/secml2/tree/main/examples).

## Contributing

We welcome contributions from the research community to expand the library's capabilities or add new features. 
If you would like to contribute to SecML2, please follow our [contribution guidelines](https://github.com/pralab/secml2/blob/main/CONTRIBUTING.md).


