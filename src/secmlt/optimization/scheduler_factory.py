"""Learning Rate Schedulers creation tools."""

from __future__ import annotations  # noqa: I001

import functools
from typing import ClassVar, TYPE_CHECKING

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

if TYPE_CHECKING:

    from torch.optim import Optimizer

LR_SCHEDULER = "lr_scheduler"
COSINE_ANNEALING = "cosine_annealing"
NO_SCHEDULER = "no_scheduler"


class NoScheduler(_LRScheduler):
    """No learning rate scheduler, does nothing."""

    def __init__(self, optimizer: Optimizer, last_epoch: int | None = -1) -> None:
        """Create a NoScheduler instance."""
        super().__init__(optimizer, last_epoch)

    def step(self, epoch: int | None = None) -> None:
        """No operation."""


class LRSchedulerFactory:
    """Creator class for learning rate schedulers."""

    SCHEDULERS: ClassVar[dict[str, _LRScheduler]] = {
        NO_SCHEDULER: NoScheduler,
        COSINE_ANNEALING: CosineAnnealingLR,
    }

    @staticmethod
    def create_from_name(
        scheduler_name: str,
        **kwargs,
    ) -> functools.partial[_LRScheduler]:
        """
        Create a learning rate scheduler.

        Parameters
        ----------
        scheduler_name : str
            One of the available scheduler names. Available: `cosine`.

        Returns
        -------
        functools.partial[LRScheduler]
            The created scheduler.

        Raises
        ------
        ValueError
            Raises ValueError when the requested scheduler is not in the list
            of implemented schedulers.
        """
        if scheduler_name == COSINE_ANNEALING:
            return LRSchedulerFactory.create_cosine_annealing()
        msg = f"Scheduler not found. Use one of: \
            {list(LRSchedulerFactory.SCHEDULERS.keys())}"
        raise ValueError(msg)

    @staticmethod
    def create_no_scheduler() -> functools.partial[_LRScheduler]:
        """
        Create a NoScheduler instance.

        Returns
        -------
        functools.partial[LRScheduler]
            NoScheduler instance.
        """
        return functools.partial(NoScheduler)

    @staticmethod
    def create_cosine_annealing() -> functools.partial[_LRScheduler]:
        """
        Create the Cosine Annealing scheduler.

        Parameters
        ----------
        TODO

        Returns
        -------
        functools.partial[LRScheduler]
            Cosine Annealing scheduler.
        """
        return functools.partial(
            CosineAnnealingLR,
        )
