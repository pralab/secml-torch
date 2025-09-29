"""Modular attack component with Expectation over Transformation (EoT) gradient."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from secmlt.models.base_model import BaseModel


class EoTGradientMixin:
    """Modular attack component with Expectation over Transformation (EoT) gradient.

    Add as a mixin to any modular attack to enable EoT gradient computation.
    """

    def __init__(
        self, eot_samples: int = 10, eot_radius: float = 0.03, *args, **kwargs
    ) -> None:
        """Add EoT gradient computation to modular attack."""
        super().__init__(*args, **kwargs)
        self.eot_samples = eot_samples
        self.eot_radius = eot_radius

    def _loss_and_grad(
        self,
        model: "BaseModel",
        samples: torch.Tensor,
        delta: torch.Tensor,
        target: torch.Tensor,
        multiplier: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute averaged finite-difference style gradient into delta.grad.

        Parameters
        ----------
        model : BaseModel
            The model to attack.
        samples : torch.Tensor
            Original clean samples.
        delta : torch.Tensor
            Current perturbation.
        target : torch.Tensor
            Target labels.
        multiplier : int
            Multiplier for loss (1 for untargeted, -1 for targeted).

        Returns
        -------
        scores : torch.Tensor
            Model scores for the adversarial examples.
        losses : torch.Tensor
            Loss values for the adversarial examples.
        """
        # basic params
        b = samples.size(0)  # batch size
        sigma = self.eot_radius  # noise scale
        k = self.eot_samples  # number of neighbors
        device = samples.device

        # ensure delta is a leaf and clear old grad
        delta.requires_grad_()
        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        # number of antithetic pairs and whether to include a center sample
        pairs = k // 2
        is_odd = (k % 2) == 1

        # prepare accumulator for losses, scores and gradient estimate
        losses_list = []
        scores_list = []
        grad_est_sum = torch.zeros_like(
            delta, device=device
        )  # gradient accumulator (w.r.t. x and thus delta)

        # vectorized pairs if pairs > 0
        if pairs > 0:
            # draw random directions for pairs
            noise_pairs = torch.randn((b, pairs, *delta.shape[1:]), device=device)

            # build pos and neg deltas of shape [b*pairs, ...]
            delta_pos = (delta.unsqueeze(1) + sigma * noise_pairs).reshape(
                b * pairs, *delta.shape[1:]
            )
            delta_neg = (delta.unsqueeze(1) - sigma * noise_pairs).reshape(
                b * pairs, *delta.shape[1:]
            )

            # expand samples and targets to match pairs
            samples_rep = (
                samples.unsqueeze(1)
                .expand(-1, pairs, *samples.shape[1:])
                .reshape(b * pairs, *samples.shape[1:])
            )
            target_rep = target.repeat_interleave(pairs, dim=0)

            # pass through manipulation function so x_pos/x_neg follow same constraints
            x_pos, _ = self.manipulation_function(samples_rep, delta_pos)
            x_neg, _ = self.manipulation_function(samples_rep, delta_neg)

            # forward for pos and neg
            pos_scores, pos_losses = self.forward_loss(
                model=model, x=x_pos, target=target_rep
            )
            neg_scores, neg_losses = self.forward_loss(
                model=model, x=x_neg, target=target_rep
            )

            # reshape back to [b, pairs] and [b, pairs, c]
            pos_losses = pos_losses.view(b, pairs)
            neg_losses = neg_losses.view(b, pairs)
            pos_scores = pos_scores.view(b, pairs, -1)
            neg_scores = neg_scores.view(b, pairs, -1)

            # collect pos/neg for averaging reporting
            losses_list.append(pos_losses)
            losses_list.append(neg_losses)
            scores_list.append(pos_scores)
            scores_list.append(neg_scores)

            # finite-difference antithetic contribution: (pos_loss - neg_loss) * noise
            # shape diffs [b, pairs] -> expand to match noise dims for broadcasting
            diffs = pos_losses - neg_losses  # [b, pairs]
            expand_dims = [1] * (
                noise_pairs.dim() - 2
            )  # e.g. for images will be [1,1] etc
            diffs_exp = diffs.view(b, pairs, *expand_dims)  # [b, pairs, 1, ...]
            contribs = diffs_exp * noise_pairs  # [b, pairs, ...]
            grad_pairs = contribs.mean(
                dim=1
            )  # average across pairs -> [b, ...] (or sum if you prefer)
            grad_est_sum += grad_pairs  # accumulate into gradient estimator

        # center sample if k is odd
        if is_odd:
            # build the center adversarial example via manipulation
            x_center, _ = self.manipulation_function(samples, delta)
            center_scores, center_losses = self.forward_loss(
                model=model, x=x_center, target=target
            )
            # add center to lists for averaging; center does not contribute to grad
            losses_list.append(center_losses.view(b, 1))
            scores_list.append(center_scores.view(b, 1, -1))

        # combine all losses and scores and compute averages for reporting
        losses_all = (
            torch.cat(losses_list, dim=1)
            if len(losses_list) > 0
            else torch.zeros((b, 0), device=device)
        )
        scores_all = (
            torch.cat(scores_list, dim=1)
            if len(scores_list) > 0
            else torch.zeros((b, 0, 0), device=device)
        )

        # multiply by multiplier and compute per-sample average loss
        avg_losses = (
            (losses_all * multiplier).mean(dim=1)
            if losses_all.numel() > 0
            else torch.zeros((b,), device=device)
        )
        avg_scores = (
            scores_all.mean(dim=1)
            if scores_all.numel() > 0
            else torch.zeros((b, 0), device=device)
        )

        # place gradient into delta.grad
        delta.grad = ((grad_est_sum / k) * multiplier).detach()

        return avg_scores.detach(), avg_losses.detach()
