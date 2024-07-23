"""Basic utils for tensor handling."""

import torch


def atleast_kd(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Add dimensions to the tensor x until it reaches k dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    k : int
        Number of desired dimensions.

    Returns
    -------
    torch.Tensor
        The input tensor x but with k dimensions.
    """
    if k <= x.dim():
        msg = "The number of desired dimensions should be > x.dim()"
        raise ValueError(msg)
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)


def normalize_l1_norm(x: torch.Tensor):
    abs_x = torch.abs(x.data)
    sorted_indices = torch.argsort(-abs_x, dim=1)
    sorted_abs_x = torch.gather(abs_x, 1, sorted_indices)
    cumsum_sorted_abs_x = torch.cumsum(sorted_abs_x, dim=1)
    mask = cumsum_sorted_abs_x <= 1.0
    mask[:, 0] = True

    selected_x = torch.where(mask, sorted_abs_x, torch.zeros_like(sorted_abs_x))
    cumsum_selected_x = torch.cumsum(selected_x, dim=1)

    residual = 1.0 - cumsum_selected_x + selected_x
    residual_mask = (mask & ~torch.roll(mask, shifts=-1, dims=1)).float()

    adjusted_x = selected_x + residual * residual_mask

    result = torch.zeros_like(x).scatter(1, sorted_indices, adjusted_x * torch.sign(x))

    return result
