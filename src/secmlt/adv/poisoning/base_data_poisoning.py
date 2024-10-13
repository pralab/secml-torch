"""Base class for data poisoning."""

import random
from typing import Union

import torch
from torch.utils.data import Dataset


class PoisoningDatasetPyTorch(Dataset):
    """Dataset class for adding poisoning samples."""

    def __init__(
        self,
        dataset: Dataset,
        data_manipulation_func: callable = lambda x: x,
        label_manipulation_func: callable = lambda x: x,
        portion: float | None = None,
        poisoned_indexes: Union[list[int], torch.Tensor] = None,
    ) -> None:
        """
        Create the poisoned dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset.
        data_manipulation_func : callable
            Function that manipulates the data.
        label_manipulation_func: callable
            Function that returns the label to associate with the poisoned data.
        portion : float, optional
            Percentage of samples on which the poisoning will be injected (default 0.1).
        poisoned_indexes: list[int] | torch.Tensor
            Specific indexes of samples to perturb. Alternative to portion.
        """
        self.dataset = dataset
        self.data_len = len(dataset)
        if portion is not None:
            if poisoned_indexes is not None:
                msg = "Specify either portion or poisoned_indexes, not both."
                raise ValueError(msg)
            if portion < 0.0 or portion > 1.0:
                msg = f"Poison ratio should be between 0.0 and 1.0. Passed {portion}."
                raise ValueError(msg)
            # calculate number of samples to poison
            num_poisoned_samples = int(portion * self.data_len)

            # randomly select indices to poison
            self.poisoned_indexes = set(
                random.sample(range(self.data_len), num_poisoned_samples)
            )
        elif poisoned_indexes is not None:
            self.poisoned_indexes = poisoned_indexes
        else:
            self.poisoned_indexes = range(self.data_len)

        self.data_manipulation_func = data_manipulation_func
        self.label_manipulation_func = label_manipulation_func

    def __len__(self) -> int:
        """Get number of samples."""
        return self.data_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
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
        # poison portion of the data
        if idx in self.poisoned_indexes:
            x = self.data_manipulation_func(x=x.unsqueeze(0)).squeeze(0)
            target_label = self.label_manipulation_func(label)
            label = (
                target_label
                if isinstance(label, int)
                else torch.Tensor(target_label).type(label.dtype)
            )
        return x, label
