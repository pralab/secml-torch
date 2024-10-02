"""Simple backdoor attack in PyTorch."""

import random
from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class BackdoorDatasetPyTorch(Dataset):
    """Dataset class for adding triggers for backdoor attacks."""

    def __init__(
        self,
        dataset: Dataset,
        trigger_label: int = 0,
        portion: float | None = None,
        poisoned_indexes: list[int] | torch.Tensor = None,
    ) -> None:
        """
        Create the backdoored dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset.
        trigger_label : int, optional
            Label to associate with the backdoored data (default 0).
        portion : float, optional
            Percentage of samples on which the backdoor will be injected (default 0.1).
        poisoned_indexes: list[int] | torch.Tensor
            Specific indexes of samples to perturb. Alternative to portion.
        """
        self.dataset = dataset
        self.trigger_label = trigger_label
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

    def add_trigger(self, x: torch.Tensor) -> torch.Tensor:
        """Modify the input by adding the backdoor."""
        return self._add_trigger(x.clone())

    @abstractmethod
    def _add_trigger(self, x: torch.Tensor) -> torch.Tensor:
        """Implement custom manipulation to add the backdoor."""

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
            x = self.add_trigger(x=x.unsqueeze(0)).squeeze(0)
            label = (
                label
                if isinstance(label, int)
                else torch.Tensor(label).type(label.dtype)
            )
        return x, label
