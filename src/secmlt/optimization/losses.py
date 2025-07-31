"""Difference of Logits Loss."""

import torch
from torch.nn.modules.loss import _WeightedLoss


class LogitDifferenceLoss(_WeightedLoss):
    """Implements the Difference of Logits Loss."""

    def __init__(self, weight: torch.Tensor = None, reduction: str = "mean") -> None:
        """Create a LogitDifferenceLoss instance."""
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the difference between input and target logits."""
        class_logits = input.amax(dim=1)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=input.size(1))
        other_logits = (input - one_hot_target).amax(dim=1)
        return class_logits - other_logits
