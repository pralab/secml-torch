from abc import abstractmethod

import torch


class Constraint:
    def __call__(self, x: torch.Tensor, *args, **kwargs):
        ...


class ClipConstraint(Constraint):
    def __init__(self, lb=0, ub=1):
        self.lb = lb
        self.ub = ub

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.clamp(self.lb, self.ub)


class LpConstraint(Constraint):
    def __init__(self, radius=0, center=0, p=torch.inf):
        self.p = p
        self.center = center
        self.radius = radius

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x + self.center
        with torch.no_grad():
            norm = torch.linalg.norm(x.flatten(start_dim=1), ord=self.p, dim=1)
            to_normalize = (norm > self.radius).view(-1, 1)
            proj_delta = self.project(x).flatten(start_dim=1)
            delta = torch.where(to_normalize, proj_delta, x.flatten(start_dim=1))
            delta = delta.view(x.shape)
        return delta

    def __repr__(self) -> str:
        return f"L{self.p} constraint with {'nonzero' if self.center!=0 else 'zero'} center with radius {self.radius}."


class L2Constraint(LpConstraint):
    def __init__(self, radius=0, center=0):
        super().__init__(radius=radius, center=center, p=2)

    def project(self, x):
        flat_x = x.flatten(start_dim=1)
        diff_norm = flat_x.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
        flat_x = flat_x / diff_norm * self.radius
        x = flat_x.reshape(x.shape)
        return x


class LInfConstraint(LpConstraint):
    def __init__(self, radius=0, center=0):
        super().__init__(radius=radius, center=center, p=float("inf"))

    def project(self, x):
        x = x.clamp(min=-self.radius, max=self.radius)
        return x


class L1Constraint(LpConstraint):
    def __init__(self, radius=0, center=0) -> None:
        super().__init__(radius=radius, center=center, p=1)

    def project(self, x):
        """
        Compute Euclidean projection onto the L1 ball for a batch.
        Source: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

        min ||x - u||_2 s.t. ||u||_1 <= eps

        Inspired by the corresponding numpy version by Adrien Gaidon.

        Parameters
        ----------
        x: (batch_size, *) torch array
        batch of arbitrary-size tensors to project, possibly on GPU

        eps: float
        radius of l-1 ball to project onto

        Returns
        -------
        u: (batch_size, *) torch array
        batch of projected tensors, reshaped to match the original

        Notes
        -----
        The complexity of this algorithm is in O(dlogd) as it involves sorting x.

        References
        ----------
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
            International Conference on Machine Learning (ICML 2008)
        """
        original_shape = x.shape
        x = x.view(x.shape[0], -1)
        mask = (torch.norm(x, p=1, dim=1) < self.radius).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, x.shape[1] + 1, device=x.device)
        rho, _ = torch.max((mu * arange > (cumsum - self.radius)) * arange, dim=1)
        theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - self.radius) / rho
        proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
        x = mask * x + (1 - mask) * proj * torch.sign(x)
        x = x.view(original_shape)
        return x
