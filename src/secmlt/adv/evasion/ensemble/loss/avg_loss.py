import torch
from torch.nn import CrossEntropyLoss


class AvgEnsembleLoss(torch.nn.Module):
    """Computes the selected loss on each ensemble model and averages them."""

    def __init__(
            self,
            loss: torch.nn.Module = CrossEntropyLoss(reduction="none"),
    ):
        """
        Creates the average ensemble loss.

        Parameters
        ----------
        loss : torch.nn.Module
            A torch Module that computes the loss. It must expose a forward
            method with the following signature:
            (input: torch.Tensor, target: torch.Tensor) -> torch.Tensor.
            Default: CrossEntropyLoss(reduction="none")
        """
        super().__init__()
        self._loss = torch.vmap(loss, in_dims=(0, None))

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the ensemble models outputs, computes the loss for each model
        and averages them.

        Parameters
        ----------
        input : torch.Tensor
            The ensemble model outputs, with shape
            (n_models, batch_size, *model_output).
        target : torch.Tensor
            The target to be used in the loss
        Returns
        -------
        torch.Tensor
            The average ensemble loss.
        """
        return self._loss(input, target).mean(dim=0)
