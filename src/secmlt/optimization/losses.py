"""Difference of Logits Loss."""

import torch
from torch.nn.modules.loss import _WeightedLoss


class LogitDifferenceLoss(_WeightedLoss):
    """Implements the Difference of Logits Loss."""

    def __init__(self, weight: torch.Tensor = None, reduction: str = "mean") -> None:
        """Create a LogitDifferenceLoss instance."""
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the difference between input and target logits.

        The loss is defined as:

        .. math::
        \mathcal{L}(x, y) = -(z_y - \max_{j \ne y} z_j)

        where:
        - :math:`z` are the model's output logits for input :math:`x`,
        - :math:`y` is the true class index,
        - :math:`z_y` is the logit corresponding to the true class,
        - :math:`\max_{j \ne y} z_j` is the highest logit among incorrect classes.

        This loss encourages the logit of the true class to be greater than
        that of any other class, and is particularly useful in adversarial
        attack settings like FMN.
        """
        # Get the true class logits (z_y)
        true_logits = input.gather(1, target.unsqueeze(1)).squeeze(1)

        # Mask out the true class by setting it to -inf
        mask = torch.ones_like(input, dtype=torch.bool)
        mask.scatter_(1, target.unsqueeze(1), value=False)
        other_logits = input.masked_fill(~mask, float("-inf")).amax(dim=1)
        return other_logits - true_logits
