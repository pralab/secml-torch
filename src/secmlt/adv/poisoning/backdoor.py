"""Simple backdoor attack in PyTorch."""

from __future__ import annotations  # noqa: I001
from typing import Union, TYPE_CHECKING

from secmlt.adv.poisoning.base_data_poisoning import PoisoningDatasetPyTorch

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset


class BackdoorDatasetPyTorch(PoisoningDatasetPyTorch):
    """Dataset class for adding triggers for backdoor attacks."""

    def __init__(
        self,
        dataset: Dataset,
        data_manipulation_func: callable,
        trigger_label: int = 0,
        portion: float | None = None,
        poisoned_indexes: Union[list[int], torch.Tensor] = None,
    ) -> None:
        """
        Create the backdoored dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset.
        data_manipulation_func: callable
            Function to manipulate the data and add the backdoor.
        trigger_label : int, optional
            Label to associate with the backdoored data (default 0).
        portion : float, optional
            Percentage of samples on which the backdoor will be injected (default 0.1).
        poisoned_indexes: list[int] | torch.Tensor
            Specific indexes of samples to perturb. Alternative to portion.
        """
        super().__init__(
            dataset=dataset,
            data_manipulation_func=data_manipulation_func,
            label_manipulation_func=lambda _: trigger_label,
            portion=portion,
            poisoned_indexes=poisoned_indexes,
        )


import torch, random
# define the backdoor dataset with cover sample
class BackdoorDatasetPyTorchWithCoverSample(PoisoningDatasetPyTorch):
    """Dataset class for adding triggers for backdoor attacks."""

    def __init__(
            self,
            dataset: Dataset,
            data_manipulation_func: callable,
            trigger_label: int = 0,
            portion: float | None = None,
            cover_portion: float = 0.0,
            poisoned_indexes: Union[list[int], torch.Tensor] = None,
    ) -> None:
        """
        Create the backdoored dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset.
        data_manipulation_func: callable
            Function to manipulate the data and add the backdoor.
        trigger_label : int, optional
            Label to associate with the backdoored data (default 0).
        portion : float, optional
            Percentage of samples on which the backdoor will be injected (default 0.1).
        poisoned_indexes: list[int] | torch.Tensor
            Specific indexes of samples to perturb. Alternative to portion.
        """
        super().__init__(
            dataset=dataset,
            data_manipulation_func=data_manipulation_func,
            label_manipulation_func=lambda _: trigger_label,
            portion=portion,
            poisoned_indexes=poisoned_indexes,
        )
        if self.poisoned_indexes is not None:
            cover_indexes_condidate = set(
                i for i in range(len(self.dataset)) if i not in self.poisoned_indexes
            )
            self.cover_indexes = set(
                random.sample(cover_indexes_condidate, int(len(self.dataset) * cover_portion))
            )
        self.weights = torch.ones(len(self.dataset))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float, int, bool]:
        """
        Get item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item to return

        Returns
        -------
        tuple[torch.Tensor, int]
            Item at position specified by idx.
        """
        x, label = self.dataset[idx]
        poison_flag = False
        # poison portion of the data
        if idx in self.poisoned_indexes:
            x = self.data_manipulation_func(x=x.unsqueeze(0)).squeeze(0)
            target_label = self.label_manipulation_func(label)
            label = (
                target_label
                if isinstance(label, int)
                else torch.Tensor(target_label).type(label.dtype)
            )
            poison_flag = True
        if idx in self.cover_indexes:
            x = self.data_manipulation_func(x=x.unsqueeze(0)).squeeze(0)

        return x, label, self.weights[idx], idx, poison_flag