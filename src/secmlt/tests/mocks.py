"""Mock classes for testing."""
import torch


class MockModel(torch.nn.Module):
    """Mock class for torch model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return random outputs for classification and add fake gradients to x."""
        # Mock output shape (batch_size, 10)
        x.grad = torch.rand_like(x)
        return torch.randn(x.size(0), 10)
