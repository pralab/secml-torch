from abc import abstractmethod

import torch


class Constraint:
    def __call__(self, x, *args, **kwargs):
        ...


class LpConstraint(Constraint):
    def __init__(self, center, radius, p):
        self.p = p
        self.center = center
        self.radius = radius

    @abstractmethod
    def _project(self, x):
        ...

    def __call__(self, x):
        x = x + self.center
        norm = torch.norm(x, p=self.p)
        if norm > self.radius:
            delta = self.project(x)
        return delta


class L2Constraint(Constraint):
    def __init__(self, center, radius):
        super().__init__(center=center, radius=radius, p=2)

    def project(self, x):
        return torch.nn.functional.Normalize(x, p=2) * self.radius


class LInfConstraint(Constraint):
    def __init__(self, center, radius):
        super().__init__(center=center, radius=radius, p=torch.int)

    def project(self, x):
        return torch.clip(x, min=-self.radius, max=self.radius)


class L1Constraint(Constraint):
    def __init__(self, center, radius) -> None:
        super().__init__(center=center, radius=radius, p=1)

    def project(self, x):
        """
        TODO fix docstring to our format
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
        return x.view(original_shape)
