import pytest
import torch
from secmlt.models.data_processing import MeanStdNormalization

MEAN = (0.5, 0.5, 0.5)
STD = (0.25, 0.25, 0.25)


@pytest.fixture
def norm():
    return MeanStdNormalization(mean=MEAN, std=STD)


@pytest.fixture
def batch():
    return torch.rand(4, 3, 8, 8)


def test_forward_shape(norm, batch):
    out = norm(batch)
    assert out.shape == batch.shape


def test_forward_values(norm, batch):
    out = norm(batch)
    mean = torch.tensor(MEAN)[None, :, None, None]
    std = torch.tensor(STD)[None, :, None, None]
    expected = (batch - mean) / std
    assert torch.allclose(out, expected)


def test_invert_roundtrip(norm, batch):
    assert torch.allclose(norm.invert(norm(batch)), batch, atol=1e-6)


def test_forward_invert_are_inverse(norm, batch):
    normalized = norm(batch)
    assert torch.allclose(norm.invert(normalized), batch, atol=1e-6)


def test_zero_mean_unit_std():
    norm = MeanStdNormalization(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    x = torch.rand(2, 3, 4, 4)
    assert torch.allclose(norm(x), x)
    assert torch.allclose(norm.invert(x), x)


def test_device_transfer(norm, batch):
    # mean/std tensors should follow the input device
    out = norm(batch.cpu())
    assert out.device == batch.device
