"""Mock classes for testing."""

from collections.abc import Iterator

import torch


class MockLayer(torch.autograd.Function):
    """Fake layer that returns the input."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        """Fake forward, returns 10 scores."""
        ctx.save_for_backward(inputs)
        return torch.randn(inputs.size(0), 10)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        """Fake backward, returns inputs."""
        (inputs,) = ctx.saved_tensors
        return inputs


class MockModel(torch.nn.Module):
    """Mock class for torch model."""

    @staticmethod
    def parameters() -> Iterator[torch.Tensor]:
        """Return fake parameters."""
        return iter([torch.rand(1, 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return random outputs for classification and add fake gradients to x."""
        # Mock output shape (batch_size, 10)
        fake_layer = MockLayer.apply
        return fake_layer(x)

    def decision_function(self, *args, **kwargs) -> torch.Tensor:
        """Return random outputs for classification and add fake gradients to x."""
        return self.forward(*args, **kwargs)
