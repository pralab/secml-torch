"""Optimizer creation tools."""

import functools
from typing import ClassVar

import torch
from torch.optim import SGD, Adam

ADAM = "adam"
StochasticGD = "sgd"


class OptimizerFactory:
    """Creator class for optimizers."""

    OPTIMIZERS: ClassVar[dict[str, torch.optim.Optimizer]] = {
        ADAM: Adam,
        StochasticGD: SGD,
    }

    @staticmethod
    def create_from_name(
        optimizer_name: str,
        lr: float,
        **kwargs,
    ) -> functools.partial[Adam] | functools.partial[SGD]:
        """
        Create an optimizer.

        Parameters
        ----------
        optimizer_name : str
            One of the available optimizer names. Available: `adam`, `sgd`.
        lr : float
            Learning rate.

        Returns
        -------
        functools.partial[Adam] | functools.partial[SGD]
            The created optimizer.

        Raises
        ------
        ValueError
            Raises ValueError when the requested optimizer is not in the list
            of implemented optimizers.
        """
        if optimizer_name == ADAM:
            return OptimizerFactory.create_adam(lr)
        if optimizer_name == StochasticGD:
            return OptimizerFactory.create_sgd(lr)
        msg = f"Optimizer not found. Use one of: \
            {list(OptimizerFactory.OPTIMIZERS.keys())}"
        raise ValueError(msg)

    @staticmethod
    def create_adam(lr: float) -> functools.partial[Adam]:
        """
        Create the Adam optimizer.

        Parameters
        ----------
        lr : float
            Learning rate.

        Returns
        -------
        functools.partial[Adam]
            Adam optimizer.
        """
        return functools.partial(Adam, lr=lr)

    @staticmethod
    def create_sgd(lr: float) -> functools.partial[SGD]:
        """
        Create the SGD optimizer.

        Parameters
        ----------
        lr : float
            Learning rate.

        Returns
        -------
        functools.partial[SGD]
            SGD optimizer.
        """
        return functools.partial(SGD, lr=lr)
