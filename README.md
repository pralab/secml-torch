 # SecML-Torch: A Library for Robustness Evaluation of Deep Learning Models

[![pypi](https://img.shields.io/pypi/v/secml-torch.svg)](https://pypi.org/pypi/secml-torch/)
[![py\_versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/pypi/secml-torch/)
[![build\_status](https://github.com/pralab/secml-torch/actions/workflows/release.yml/badge.svg)](https://github.com/pralab/secml-torch/actions?query=workflow%3ACI)
[![coverage](https://codecov.io/gh/pralab/secml-torch/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pralab/secml-torch)
[![docs](https://readthedocs.org/projects/secml-torch/badge/?version=latest)](https://secml-torch.readthedocs.io/en/latest/#)

SecML-Torch (SecMLT) is an open-source Python library designed to facilitate research in the area of Adversarial Machine Learning (AML) and robustness evaluation.
The library provides a simple yet powerful interface for generating various types of adversarial examples, as well as tools for evaluating the robustness of machine learning models against such attacks.

## Installation

You can install SecMLT via pip:
```bash
pip install secml-torch
```

This will install the core version of SecMLT, including only the main functionalities such as native implementation of attacks and PyTorch wrappers.

### Install with extras

The library can be installed together with other plugins that enable further functionalities.

* [Foolbox](https://github.com/bethgelab/foolbox), a Python toolbox to create adversarial examples.
* [Tensorboard](https://www.tensorflow.org/tensorboard), a visualization toolkit for machine learning experimentation.

Install one or more extras with the command:
```bash
pip install secml-torch[foolbox,tensorboard]
```

## Key Features

- **Built for Deep Learning:** SecMLT is compatible with the popular machine learning framework PyTorch.
- **Various types of adversarial attacks:** SecMLT includes support for a wide range of attack methods (evasion, poisoning, ...) such as different implementations imported from popular AML libraries (Foolbox, Adversarial Library).
- **Customizable attacks:** SecMLT offers several levels of analysis for the models, including modular implementations of existing attacks to extend with different loss functions, optimizers, and more.
- **Attack debugging:** Built-in debugging of evaluations by logging events and metrics along the attack runs (even on Tensorboard).

## Usage

Here's a brief example of using SecMLT to evaluate the robustness of a trained classifier:

```python
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


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

For more detailed usage instructions and examples, please refer to the [official documentation](https://secml-torch.readthedocs.io/en/latest/) or to the [examples](https://github.com/pralab/secml-torch/tree/main/examples).

## Contributing

We welcome contributions from the research community to expand the library's capabilities or add new features.
If you would like to contribute to SecMLT, please follow our [contribution guidelines](https://github.com/pralab/secml-torch/blob/main/CONTRIBUTING.md).

### Contributors

<table>
<tr>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/maurapintor>
            <img src=https://avatars.githubusercontent.com/u/23276849?v=4 width="50;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding:10px" alt=Maura Pintor/>
            <br />
            <sub style="font-size:14px;padding:10px"><b>maurapintor</b></sub>
        </a>
    </td>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/zangobot>
            <img src=https://avatars.githubusercontent.com/u/6831380?v=4 width="50;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding:10px" alt=zangobot/>
            <br />
            <sub style="font-size:14px;padding:10px"><b>zangobot</b></sub>
        </a>
    </td>
</tr>
</table>
