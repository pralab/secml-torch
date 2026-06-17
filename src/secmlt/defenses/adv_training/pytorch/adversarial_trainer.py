r"""Adversarial Training Class for PyTorch models.

The adversarial training implements the method of Madry et al. (2017) to solve a min-max
optimization problem, where the inner maximization is solved by generating adversarial
examples with a given attack, and the outer minimization is solved by training the model
(only) on adversarial examples.
"""

from collections.abc import Iterator
from typing import Literal, Union

import torch
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.nn import Module
from torch.utils.data import DataLoader

COMBINING_MODE = Literal["adv", "mix"]


class AdversarialTrainer(BasePyTorchTrainer):
    """Adversarial trainer for PyTorch models."""

    def train(
        self,
        model: Union[BasePyTorchClassifier, Module],
        dataloader: DataLoader,
        attack: BaseEvasionAttack,
        combining_mode: COMBINING_MODE = "adv",
    ) -> Module:
        r"""Train a model with the given dataloader and attack.

        Parameters
        ----------
        model : BasePyTorchClassifier or Module
            Model to train. Can be a raw ``torch.nn.Module`` or a
            ``BasePyTorchClassifier`` wrapper.
        dataloader : DataLoader
            Training dataloader.
        attack : BaseEvasionAttack
            Adversarial attack to use for adversarial training.
        combining_mode : COMBINING_MODE, optional
            Strategy to combine the original and adversarial dataloaders. Options are:
            - "adv": Use only adversarial examples.
            - "mix": Mix original and adversarial examples.

        Returns
        -------
        Module
            The trained underlying ``torch.nn.Module``, set to eval mode.
        """
        if not isinstance(model, BasePyTorchClassifier):
            model = BasePyTorchClassifier(model)

        for _ in range(self._epochs):
            # Generate adversarial examples using the attack and the dataloader
            adv_data = attack(model, dataloader, stream=True)

            # Combine the original and adversarial dataloaders
            combined_data = self.collect_data(dataloader, adv_data, combining_mode)

            # Train through the full classifier so preprocessing (e.g. input
            # normalization) is applied consistently with the attack step.
            model.model.train()
            self._train_epoch_clf(model, combined_data)
            if self._scheduler is not None:
                self._scheduler.step()
        # Restore eval mode so the returned model is ready for inference and
        # its BatchNorm layers use running statistics instead of batch ones.
        return model.model.eval()

    def _train_epoch_clf(
        self, classifier: BasePyTorchClassifier, dataloader: DataLoader
    ) -> None:
        """Run one training epoch using the full classifier forward pass."""
        param = next(iter(classifier.model.parameters()), None)
        device = param.device if param is not None else torch.device("cpu")
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self._optimizer.zero_grad()
            outputs = classifier.decision_function(x)
            loss = self._loss(outputs, y)
            loss.backward()
            self._optimizer.step()

    def collect_data(
        self,
        clean_data: DataLoader,
        adv_data: Iterator,
        combining_strategy: COMBINING_MODE,
    ) -> Iterator:
        r"""Combine two data sources into one.

        Parameters
        ----------
        clean_data : DataLoader
        adv_data : Iterator
        combining_strategy : COMBINING_MODE
            Strategy to combine the dataloaders. Options are:
            - "adv": Use only adversarial examples.
            - "mix": Mix original and adversarial examples.

        Returns
        -------
        Iterator
            Combined data source containing samples from both input data sources.
        """
        if combining_strategy == "adv":
            return adv_data
        if combining_strategy == "mix":
            err_msg = "Mixing strategy not implemented yet."
            raise NotImplementedError(err_msg)
        err_msg = (
            f"Invalid combining_strategy: {combining_strategy}. "
            "Supported values are 'adv' and 'mix'."
        )
        raise ValueError(err_msg)
