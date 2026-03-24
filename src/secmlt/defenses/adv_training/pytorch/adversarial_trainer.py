r"""Adversarial Training Class for PyTorch models.

The adversarial training implements the method of Madry et al. (2017) to solve a min-max
optimization problem, where the inner maximization is solved by generating adversarial
examples with a given attack, and the outer minimization is solved by training the model
(only) on adversarial examples.
"""
from typing import Literal

from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.nn import Module
from torch.utils.data import DataLoader

COMBINING_MODE = Literal["adv", "mix"]

class AdversarialTrainer(BasePyTorchTrainer):
    """Adversarial trainer for PyTorch models."""

    def train(self, model: Module,
              dataloader: DataLoader,
              attack: BaseEvasionAttack,
              combining_mode: COMBINING_MODE = "adv") -> Module:
        r"""Train a model with the given dataloader and attack.

        Parameters
        ----------
        model : Module
            Model to train.
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
            The trained model.
        """
        for _ in range(self._epochs):
            # Generate adversarial examples using the attack and the dataloader
            model.eval()
            adv_data = attack(BasePyTorchClassifier(model), dataloader)

            # Combine the original and adversarial dataloaders
            combined_data = self.collect_data(dataloader, adv_data, combining_mode)

            # Train the model using the combined dataloader
            model = model.train()
            model = self.train_epoch(model, combined_data)
            if self._scheduler is not None:
                self._scheduler.step()
        return model

    def collect_data(self, clean_data: DataLoader, adv_data: DataLoader,
                     combining_strategy: COMBINING_MODE) -> DataLoader:
        r"""Combine two dataloaders into one.

        Parameters
        ----------
        clean_data : DataLoader
        adv_data : DataLoader
        combining_strategy : COMBINING_MODE
            Strategy to combine the dataloaders. Options are:
            - "adv": Use only adversarial examples.
            - "mix": Mix original and adversarial examples.

        Returns
        -------
        DataLoader
            Combined dataloader containing samples from both input dataloaders.
        """
        if combining_strategy == "adv":
            return adv_data
        if combining_strategy == "mix":
            err_msg = "Mixing strategy not implemented yet."
            raise NotImplementedError(err_msg)
        return None
