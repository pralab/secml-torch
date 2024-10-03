"""Simple backdoor attack in PyTorch."""

import random

import torch
from secmlt.adv.poisoning.base_data_poisoning import PoisoningDatasetPyTorch
from torch.utils.data import Dataset


class BackdoorDatasetPyTorch(PoisoningDatasetPyTorch):
    """Dataset class for adding triggers for backdoor attacks."""

    def __init__(
        self,
        dataset: Dataset,
        data_manipulation_func: callable,
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
        data_manipulation_func: callable
            Function to manipulate the data and add the backdoor.
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
        self.data_manipulation_func = data_manipulation_func
        self.label_manipulation_func = lambda _: trigger_label
