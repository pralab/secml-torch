import torch


class Manipulation:
    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract manipulation.")

    def invert(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract inversion.")


class AdditiveManipulation(Manipulation):
    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_adv = x + delta
        return x_adv

    def invert(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        delta = x_adv - x
        return delta
