import torch


def atleast_kd(x: torch.Tensor, k: int) -> torch.Tensor:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)
