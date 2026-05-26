"""Classification metrics for machine-learning models and for attack performance."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import torch
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from secmlt.models.base_language_model import BaseLanguageModel


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy on a batch of predictions and targets.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions from the model.
    y_true : torch.Tensor
        Target labels.

    Returns
    -------
    torch.Tensor
        The percentage of predictions that match the targets.
    """
    return (y_pred.type(y_true.dtype) == y_true).float().mean()


class Accuracy:
    """Class for computing accuracy of a model on a dataset."""

    def __init__(self) -> None:
        """Create Accuracy metric."""
        self._num_samples = 0
        self._accumulated_accuracy = 0.0

    def __call__(self, model: BaseModel, dataloader: DataLoader) -> torch.Tensor:
        """
        Compute the metric on a single attack run or a dataloader.

        Parameters
        ----------
        model : BaseModel
            Model to use for prediction.
        dataloader : DataLoader
            A dataloader, can be the result of an attack or a generic
            test dataloader.

        Returns
        -------
        torch.Tensor
            The metric computed on the given dataloader.
        """
        for _, (x, y) in enumerate(dataloader):
            y_pred = model.predict(x).cpu().detach()
            self._accumulate(y_pred, y)
        return self._compute()

    def _accumulate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self._num_samples += y_true.shape[0]
        self._accumulated_accuracy += torch.sum(
            y_pred.type(y_true.dtype).cpu() == y_true.cpu(),
        )

    def _compute(self) -> torch.Tensor:
        return self._accumulated_accuracy / self._num_samples


class AttackSuccessRate(Accuracy):
    """Single attack success rate from attack results."""

    def __init__(self, y_target: Union[float, torch.Tensor, None] = None) -> None:
        """
        Create attack success rate metric.

        Parameters
        ----------
        y_target : float | torch.Tensor | None, optional
            Target label for the attack, None for untargeted, by default None
        """
        super().__init__()
        self.y_target = y_target

    def _accumulate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        if self.y_target is None:
            super()._accumulate(y_pred, y_true)
        else:
            super()._accumulate(y_pred, torch.ones_like(y_true) * self.y_target)

    def _compute(self) -> torch.Tensor:
        if self.y_target is None:
            return 1 - super()._compute()
        return super()._compute()


class AccuracyEnsemble(Accuracy):
    """Robust accuracy of a model on multiple attack runs."""

    def __call__(self, model: BaseModel, dataloaders: list[DataLoader]) -> torch.Tensor:
        """
        Compute the metric on an ensemble of attacks from their results.

        Parameters
        ----------
        model : BaseModel
            Model to use for prediction.
        dataloaders : list[DataLoader]
            List of loaders returned from multiple attack runs.

        Returns
        -------
        torch.Tensor
            The metric computed across multiple attack runs.
        """
        for advs in zip(*dataloaders):  # noqa: B905
            y_pred = []
            for x, y in advs:
                y_pred.append(model.predict(x).cpu().detach())
                # verify that the samples order correspond
                assert (y - advs[0][1]).sum() == 0
            y_pred = torch.vstack(y_pred)
            self._accumulate(y_pred, advs[0][1])
        return self._compute()

    def _accumulate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self._num_samples += y_true.shape[0]
        self._accumulated_accuracy += torch.sum(
            # take worst over predictions
            (y_pred.type(y_true.dtype).cpu() == y_true.cpu()).min(dim=0).values,
        )


class EnsembleSuccessRate(AccuracyEnsemble):
    """Worst-case success rate of multiple attack runs."""

    def __init__(self, y_target: Union[float, torch.Tensor, None] = None) -> None:
        """
        Create ensemble success rate metric.

        Parameters
        ----------
        y_target : float | torch.Tensor | None, optional
            Target label for the attack, None for untargeted,, by default None
        """
        super().__init__()
        self.y_target = y_target

    def _accumulate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        if self.y_target is None:
            super()._accumulate(y_pred, y_true)
        else:
            self._num_samples += y_true.shape[0]
            self._accumulated_accuracy += torch.sum(
                # take worst over predictions
                (
                    y_pred.type(y_true.dtype).cpu()
                    == (torch.ones_like(y_true) * self.y_target).cpu()
                )
                .max(dim=0)
                .values,
            )

    def _compute(self) -> torch.Tensor:
        if self.y_target is None:
            return 1 - super()._compute()
        return super()._compute()


class JailbreakAccuracy(Accuracy):
    """Accuracy of a jailbreak success criterion."""

    def __init__(
        self,
        is_jailbreak: Callable[
            ["BaseLanguageModel", dict[str, Any], str, dict[str, Any]],
            bool,
        ],
    ) -> None:
        """
        Create a jailbreak accuracy metric.

        Parameters
        ----------
        is_jailbreak : callable
            Function returning True when an adversarial prompt is a jailbreak.
            This can wrap a judge model, string matcher, logit probe, or
            latent-space probe.
        """
        super().__init__()
        self.is_jailbreak = is_jailbreak

    def __call__(
        self,
        model: "BaseLanguageModel",
        behaviors: list[dict[str, Any]],
        attack_results: list[tuple[str, dict[str, Any]]],
    ) -> torch.Tensor:
        """
        Compute jailbreak accuracy on a batch of attack results.

        Parameters
        ----------
        model : BaseLanguageModel
            Victim language model.
        behaviors : list of dict
            Original behavior specifications.
        attack_results : list of tuples
            List of (adv_prompt, logs) pairs returned by a jailbreak attack.

        Returns
        -------
        torch.Tensor
            Fraction of prompts judged as successful jailbreaks.
        """
        if len(behaviors) != len(attack_results):
            msg = "Behaviors and attack results must have the same length."
            raise ValueError(msg)
        if len(behaviors) == 0:
            msg = "Cannot compute jailbreak accuracy on an empty batch."
            raise ValueError(msg)

        y_pred = torch.tensor(
            [
                self.is_jailbreak(model, behavior, adv_prompt, logs)
                for behavior, (adv_prompt, logs) in zip(
                    behaviors, attack_results, strict=True
                )
            ],
            dtype=torch.bool,
        )
        y_true = torch.ones_like(y_pred)
        self._accumulate(y_pred, y_true)
        return self._compute()
