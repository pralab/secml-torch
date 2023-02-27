import torch


class Manipulation:
    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract manipulation.")


class AdditiveManipulation(Manipulation):
    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_adv = x + delta
        return x_adv
