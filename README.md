  <p align="center">
  <img src="_static/assets/logos/logo_horizontal.png" alt=secml-torch style="width:500px;"/> &nbsp;&nbsp;
</p>

 # SecML-Torch: A Library for Robustness Evaluation of Deep Learning Models

[![pypi](https://img.shields.io/badge/pypi-latest-blue)](https://pypi.org/pypi/secml-torch/)
[![py\_versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/pypi/secml-torch/)
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
* [Adversarial Library](https://github.com/jeromerony/adversarial-library), a powerful library of various adversarial attacks resources in PyTorch.


Install one or more extras with the command:
```bash
pip install secml-torch[foolbox,tensorboard,adv_lib]
```

## Key Features

SecML-Torch (SecMLT) is a PyTorch-native toolkit for evaluating and improving adversarial robustness. It provides:
- Built-in support for evaluating PyTorch models.
- **Efficient native implementations** of common evasion attacks (e.g. PGD, FMN), built directly for PyTorch, and poisoning/backdoor attacks.
- **Wrappers for external libraries** (Foolbox, Adversarial Library) so you can run and compare attacks from a single interface.
- **Modular design to build adaptive/custom attacks** to swap losses, optimizers, perturbation models, and add EoT easily with a few lines of code.
- **Attack ensembling modules** to obtain worst-case per-sample robustness evaluations.
- **Robustness evaluation tools** including metrics, logging, and trackers to ensure reproducibility and easy reporting.
- **Attacl debugging support** such as built-in hooks and TensorBoard integration to monitor and inspect attack behavior.

Check out the [tutorials](https://secml-torch.readthedocs.io) to see SecML-Torch in action.

| Category | Attack / Attack Type | Native Implementation in SecML-Torch | Wrapped / Imported / Backend Alternatives |
| - | - | - | - |
|  |  | <sub>Advantages: GPU-native, efficient, modular/customizable, debugging tools</sub> | <sub>Advantages: expands the attack catalogue, easy cross-checks</sub> |
| Test time | **PGD (fixed-epsilon, iterative attack)** | ✔ Native implementation | ✔ Also via backend wrappers (Foolbox, Adversarial Library). |
| Test time | **FMN (minimum-norm, iterative attack)** | ✔ Native implementation | ✔ Also via backend wrappers (Foolbox, Adversarial Library). |
| Test time | **DDN (minimum-norm, iterative attack)** | ✔ Native implementation | ✔ Also via backend wrappers (Foolbox, Adversarial Library). |
| Test time | **Other Evasion Attacks** | Work in progress | ✔ Available via backend wrappers (Foolbox, Adversarial Library). |
| Training time | **Backdoor** | ✔ Native implementation | - |
| Training time | **Label Flip Poisoning** | ✔ Native implementation | - |

Check out what's cooking! Have a look at our [roadmap](https://github.com/orgs/pralab/projects/2)!

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


## Acknowledgements
SecML has been partially developed with the support of European Union’s [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), Horizon Europe, grant agreement No. 101070617, [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu), Horizon Europe, grant agreement No. 101120393, and [CoEvolution - A Comprehensive Trustworthy Framework for Connected Machine Learning and Secure Interconnected AI Solutions](https://coevolution-project.eu/), Horizon Europe, grant agreement No. 101168560, and by the project SERICS (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union - NextGenerationEU.

<img src="_static/assets/logos/sec4AI4sec.png" alt="sec4ai4sec" style="height:60px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/elsa.jpg" alt="elsa" style="height:60px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/coevolution.svg" alt="coevolution" style="height:60px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/serics.png" alt="serics" style="height:60px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/FundedbytheEU.png" alt="europe" style="height:60px;"/>
