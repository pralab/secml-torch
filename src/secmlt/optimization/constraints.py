"""Constraints for tensors and the corresponding batch-wise projections."""

from abc import ABC, abstractmethod

import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels


class Constraint(ABC):
    """Generic constraint."""

    @abstractmethod
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Project onto the constraint.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor projected onto the constraint.
        """
        ...


class ClipConstraint(Constraint):
    """Box constraint, usually for the input space."""

    def __init__(self, lb: float = 0.0, ub: float = 1.0) -> None:
        """
        Create box constraint.

        Parameters
        ----------
        lb : float, optional
            Lower bound of the domain, by default 0.0.
        ub : float, optional
            Upper bound of the domain, by default 1.0.
        """
        self.lb = lb
        self.ub = ub

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Call the projection function.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Tensor projected onto the box constraint.
        """
        return x.clamp(self.lb, self.ub)


class LpConstraint(Constraint, ABC):
    """Abstract class for Lp constraint."""

    def __init__(
        self,
        radius: float = 0.0,
        center: float = 0.0,
        p: str = LpPerturbationModels.LINF,
    ) -> None:
        """
        Create Lp constraint.

        Parameters
        ----------
        radius : float, optional
            Radius of the constraint, by default 0.0.
        center : float, optional
            Center of the constraint, by default 0.0.
        p : str, optional
            Value of p for Lp norm, by default LpPerturbationModels.LINF.
        """
        self.p = LpPerturbationModels.get_p(p)
        self.center = center
        self.radius = radius

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto the Lp constraint.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor projected onto the Lp constraint.
        """
        ...

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Project the samples onto the Lp constraint.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor projected onto the Lp constraint.
        """
        x = x + self.center
        with torch.no_grad():
            norm = torch.linalg.norm(x.flatten(start_dim=1), ord=self.p, dim=1)
            to_normalize = (norm > self.radius).view(-1, 1)
            proj_delta = self.project(x).flatten(start_dim=1)
            delta = torch.where(to_normalize, proj_delta, x.flatten(start_dim=1))
        return delta.view(x.shape)


class L2Constraint(LpConstraint):
    """L2 constraint."""

    def __init__(self, radius: float = 0.0, center: float = 0.0) -> None:
        """
        Create L2 constraint.

        Parameters
        ----------
        radius : float, optional
            Radius of the constraint, by default 0.0.
        center : float, optional
            Center of the constraint, by default 0.0.
        """
        super().__init__(radius=radius, center=center, p=LpPerturbationModels.L2)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto the L2 constraint.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor projected onto the L2 constraint.
        """
        flat_x = x.flatten(start_dim=1)
        diff_norm = flat_x.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
        flat_x = torch.where(diff_norm <= 1, flat_x, flat_x / diff_norm) * self.radius
        return flat_x.reshape(x.shape)


class LInfConstraint(LpConstraint):
    """Linf constraint."""

    def __init__(self, radius: float = 0.0, center: float = 0.0) -> None:
        """
        Create Linf constraint.

        Parameters
        ----------
        radius : float, optional
            Radius of the constraint, by default 0.0.
        center : float, optional
            Center of the constraint, by default 0.0.
        """
        super().__init__(radius=radius, center=center, p=LpPerturbationModels.LINF)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto the Linf constraint.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor projected onto the Linf constraint.
        """
        return x.clamp(min=-self.radius, max=self.radius)


class L1Constraint(LpConstraint):
    """L1 constraint."""

    def __init__(self, radius: float = 0.0, center: float = 0.0) -> None:
        """
        Create L1 constraint.

        Parameters
        ----------
        radius : float, optional
            Radius of the constraint, by default 0.0.
        center : float, optional
            Center of the constraint, by default 0.0.
        """
        super().__init__(radius=radius, center=center, p=1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean projection onto the L1 ball for a batch.

        Source: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

        min ||x - u||_2 s.t. ||u||_1 <= eps

        Inspired by the corresponding numpy version by Adrien Gaidon.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected tensor.

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


class L0Constraint(LpConstraint):
    """L0 constraint."""

    def __init__(self, radius: float = 0.0, center: float = 0.0) -> None:
        """
        Create L0 constraint.

        Parameters
        ----------
        radius : float, optional
            Radius of the constraint, by default 0.0.
        center : float, optional
            Center of the constraint, by default 0.0.
        """
        super().__init__(radius=radius, center=center, p=0)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the samples onto the L0 constraint.

        Returns the sample with the top-k components preserved,
        and the rest set to zero.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Samples projected onto L0 constraint.
        """
        flat_x = x.flatten(start_dim=1)
        positions, topk = torch.topk(flat_x, k=self.radius)
        return torch.zeros_like(flat_x).scatter_(positions, topk).reshape(x.shape)
