"""Simple backdoor attack in PyTorch."""

from typing import Union

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
