from __future__ import annotations

import pytest
import torch
from secmlt.optimization import (
    ClipConstraint,
    L0Constraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
    MaskConstraint,
    QuantizationConstraint,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def assert_tensor_equal(actual, expected, msg="") -> None:
    assert torch.allclose(actual, expected), (
        f"Expected {expected}, but got {actual}. {msg}"
    )


class _LargestTensor(TorchDispatchMode):
    """Record the largest number of elements of any tensor produced by an op."""

    def __init__(self) -> None:
        super().__init__()
        self.numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                self.numel = max(self.numel, t.numel())
        return out


@pytest.mark.parametrize(
    "lb, ub, x, expected",
    [
        (
            0.0,
            1.0,
            torch.tensor([[1.5, -0.5], [0.3, 0.7]]),
            torch.tensor([[1.0, 0.0], [0.3, 0.7]]),
        ),
        (
            -1.0,
            1.0,
            torch.tensor([[1.5, -1.5], [0.3, 0.7]]),
            torch.tensor([[1.0, -1.0], [0.3, 0.7]]),
        ),
    ],
)
def test_clip_constraint(lb, ub, x, expected):
    constraint = ClipConstraint(lb=lb, ub=ub)
    projected = constraint(x)
    assert_tensor_equal(projected, expected)


@pytest.mark.parametrize(
    "constraint_class, radius, x, expected_norm, norm_type",
    [
        (L2Constraint, 5.0, torch.tensor([[3.0, 4.0], [0.5, 0.5]]), 5.0, 2),
        (
            LInfConstraint,
            1.0,
            torch.tensor([[3.0, -4.0], [0.5, 0.5]]),
            1.0,
            float("inf"),
        ),
        (L1Constraint, 2.0, torch.tensor([[2.0, 2.0], [1.0, -1.0]]), 2.0, 1),
    ],
)
def test_lp_constraints(constraint_class, radius, x, expected_norm, norm_type):
    constraint = constraint_class(radius=radius)
    projected = constraint(x)

    # calculate the norm for the projection and ensure it does not exceed the radius
    norms = torch.norm(projected.flatten(start_dim=1), p=norm_type, dim=1)
    assert torch.all(norms <= expected_norm), (
        f"{constraint_class.__name__} failed with norms {norms}"
    )


@pytest.mark.parametrize(
    "x, k, expected",
    [
        (
            torch.tensor([[-1.5, -0.5, 0.1, 2.0], [0.3, 0.7, 0.1, 1.0]]),
            2,
            torch.tensor([[-1.5, 0.0, 0.0, 2.0], [0.0, 0.7, 0.0, 1.0]]),
        )
    ],
)
def test_l0_constraint(x, k, expected):
    constraint = L0Constraint(radius=k)
    projected = constraint(x)

    # top 2 values should remain, others should be zero
    assert_tensor_equal(projected, expected)


def test_l0_constraint_invalid_radius():
    # Test that passing a non-integer radius raises an error
    with pytest.raises(ValueError):  # noqa: PT011
        L0Constraint(radius=2.5)


@pytest.mark.parametrize(
    "x, kwargs, expected",
    [
        (
            torch.tensor([[0.2, 0.7], [0.4, 0.8]]),
            {"levels": 5},
            torch.tensor([[0.25, 0.75], [0.5, 0.75]]),
        ),
        (
            torch.tensor([[0.1, 0.9], [0.3, 0.6]]),
            {"bins": torch.Tensor([0.1, 0.2, 0.5])},
            torch.tensor([[0.1, 0.5], [0.2, 0.5]]),
        ),
        (
            torch.tensor([[0.1, 0.9], [0.3, 0.6]]),
            {"bins": [0.1, 0.2, 0.5]},
            torch.tensor([[0.1, 0.5], [0.2, 0.5]]),
        ),
    ],
)
def test_quantization_constraint(x, kwargs, expected):
    constraint = QuantizationConstraint(**kwargs)
    projected = constraint(x)
    assert_tensor_equal(projected, expected)


def test_quantization_constraint_default_levels():
    # without arguments, the constraint uses 255 equally spaced levels
    x = torch.rand(100, generator=torch.Generator().manual_seed(0))
    assert torch.equal(
        QuantizationConstraint()(x), QuantizationConstraint(levels=255)(x)
    )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"levels": 5, "bins": [0.0, 1.0]}, ValueError),
        ({"levels": 1}, ValueError),
        ({"levels": 2.5}, TypeError),
        ({"levels": True}, TypeError),
        ({"levels": [0.0, 1.0]}, TypeError),
        ({"levels": torch.tensor(5)}, TypeError),
        ({"bins": [0.5]}, ValueError),
        ({"bins": []}, ValueError),
        ({"bins": torch.tensor(0.5)}, ValueError),
    ],
)
def test_quantization_constraint_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        QuantizationConstraint(**kwargs)


def _nearest_level_reference(x, bins):
    # brute-force nearest value (ties go to the lower value), used as reference
    bins = torch.sort(torch.as_tensor(bins, dtype=torch.float32)).values
    distances = torch.abs(x.unsqueeze(-1) - bins)
    return bins[torch.argmin(distances, dim=-1)]


@pytest.mark.parametrize("kwargs", [{"levels": 5}, {"bins": [0, 0.25, 0.5, 0.75, 1]}])
def test_quantization_constraint_ties_go_to_lower_level(kwargs):
    # exact midpoints between two values are quantized to the lower one
    x = torch.tensor([0.125, 0.375, 0.625, 0.875])
    projected = QuantizationConstraint(**kwargs)(x)
    assert_tensor_equal(projected, torch.tensor([0.0, 0.25, 0.5, 0.75]))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"levels": 3}, torch.tensor([0.0, 1.0])),
        ({"bins": [0.2, 0.8]}, torch.tensor([0.2, 0.8])),
    ],
)
def test_quantization_constraint_out_of_range(kwargs, expected):
    # values outside the range go to the closest end value
    projected = QuantizationConstraint(**kwargs)(torch.tensor([-1.0, 2.0]))
    assert_tensor_equal(projected, expected)


def test_quantization_constraint_unsorted_bins():
    x = torch.tensor([0.1, 0.4, 0.6, 0.9])
    unsorted = QuantizationConstraint(bins=[0.5, 0.0, 1.0])(x)
    ordered = QuantizationConstraint(bins=[0.0, 0.5, 1.0])(x)
    assert torch.equal(unsorted, ordered)


@pytest.mark.parametrize(
    "bins",
    [
        [0.0, 0.5, 0.5, 1.0],
        [0.1, 0.2, 0.5],
        torch.sort(torch.rand(37, generator=torch.Generator().manual_seed(1))).values,
    ],
)
def test_quantization_constraint_matches_reference(bins):
    # custom bins select exactly the nearest value, also out of range
    x = torch.rand(10_000, generator=torch.Generator().manual_seed(0)) * 1.4 - 0.2
    projected = QuantizationConstraint(bins=bins)(x)
    assert torch.equal(projected, _nearest_level_reference(x, bins))


@pytest.mark.parametrize(
    "kwargs",
    [{"levels": 2}, {"levels": 255}, {"levels": 1000}, {"bins": [0.1, 0.2, 0.5]}],
)
def test_quantization_constraint_idempotent(kwargs):
    # quantizing already quantized values does not change them
    x = torch.rand(10_000, generator=torch.Generator().manual_seed(0)) * 1.4 - 0.2
    constraint = QuantizationConstraint(**kwargs)
    projected = constraint(x)
    assert torch.equal(constraint(projected), projected)


@pytest.mark.parametrize(
    "kwargs", [{"levels": 255}, {"bins": torch.linspace(0, 1, 255) ** 2}]
)
def test_quantization_constraint_memory(kwargs):
    # no intermediate tensor grows with the number of levels (issue #202)
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    constraint = QuantizationConstraint(**kwargs)
    bins_numel = 0 if constraint.bins is None else constraint.bins.numel()
    with _LargestTensor() as tracker:
        constraint(x)
    assert tracker.numel <= max(x.numel(), bins_numel)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("levels", [2, 3, 16, 255, 256, 1000])
def test_quantization_constraint_uniform_levels(levels, dtype):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(10_000, generator=generator, dtype=dtype) * 1.4 - 0.2
    constraint = QuantizationConstraint(levels=levels)
    projected = constraint(x)
    step = 1 / (levels - 1)
    tolerance = 10 * torch.finfo(dtype).eps
    assert projected.dtype == dtype
    assert (projected - x.clamp(0, 1)).abs().max() <= step / 2 + tolerance
    assert not torch.signbit(projected).any()
    assert torch.equal(constraint(projected), projected)
    grid = torch.arange(levels, dtype=dtype) / (levels - 1)
    assert torch.equal(constraint(grid), grid)
    # away from the midpoints between two levels, the nearest level is unambiguous
    scaled = x.clamp(0, 1) * (levels - 1)
    away = (scaled - scaled.floor() - 0.5).abs() > 1e-3  # noqa: PLR2004
    reference = _nearest_level_reference(x, torch.linspace(0, 1, levels))
    assert torch.allclose(projected[away].float(), reference[away], atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"levels": 3}, {"bins": [0.0, 0.5, 1.0]}])
def test_quantization_constraint_integer_input(kwargs):
    # integer inputs are quantized as floats, without truncating the values
    projected = QuantizationConstraint(**kwargs)(torch.tensor([[0, 1]]))
    assert projected.dtype == torch.float32
    assert torch.equal(projected, torch.tensor([[0.0, 1.0]]))


def _per_feature_reference(x, bins):
    # nearest value computed separately for each feature
    flat_x = x.reshape(x.shape[0], -1)
    flat_bins = bins.reshape(-1, bins.shape[-1])
    columns = [
        _nearest_level_reference(flat_x[:, f], flat_bins[f])
        for f in range(flat_bins.shape[0])
    ]
    return torch.stack(columns, dim=1).reshape(x.shape)


@pytest.mark.parametrize("shape", [(64, 3), (5, 3, 4, 4)])
def test_quantization_constraint_per_feature_bins(shape):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(shape, generator=generator) * 1.4 - 0.2
    bins = torch.rand(*shape[1:], 6, generator=generator)  # unsorted on purpose
    projected = QuantizationConstraint(bins=bins)(x)
    assert projected.shape == x.shape
    assert torch.equal(projected, _per_feature_reference(x, bins))


def test_quantization_constraint_per_feature_bins_ties():
    # each feature uses its own values, ties go to the lower one
    bins = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.5, 1.0, 1.0, 1.0]])
    projected = QuantizationConstraint(bins=bins)(torch.tensor([[0.125, 0.25]]))
    assert torch.equal(projected, torch.tensor([[0.0, 0.0]]))


def test_quantization_constraint_per_feature_bins_wrong_shape():
    constraint = QuantizationConstraint(bins=torch.rand(4, 6))
    with pytest.raises(ValueError):  # noqa: PT011
        constraint(torch.rand(2, 3))


def test_quantization_constraint_per_feature_bins_memory():
    # nothing scales with batch size times number of values
    x = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    constraint = QuantizationConstraint(bins=torch.rand(3, 16, 16, 255))
    with _LargestTensor() as tracker:
        constraint(x)
    assert tracker.numel <= max(x.numel(), constraint.bins.numel())


def test_quantization_constraint_bins_keep_dtype():
    x = torch.tensor([[0.1, 0.3, 0.9]], dtype=torch.float64)
    projected = QuantizationConstraint(bins=[0.0, 0.5, 1.0])(x)
    assert projected.dtype == torch.float64
    assert torch.equal(projected, torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64))


def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.mark.skipif(_accelerator() is None, reason="no cuda or mps device")
@pytest.mark.parametrize("kwargs", [{"levels": 5}, {"bins": [0.0, 0.25, 1.0]}])
def test_quantization_constraint_device(kwargs):
    # bins are moved to the input device
    x = torch.tensor([[0.1, 0.3, 0.9]], device=_accelerator())
    projected = QuantizationConstraint(**kwargs)(x)
    assert projected.device == x.device
    assert torch.equal(projected.cpu(), QuantizationConstraint(**kwargs)(x.cpu()))


@pytest.mark.parametrize(
    "x, mask, expected",
    [
        (
            torch.tensor([[0.2, 0.7], [0.4, 0.8]]),
            torch.tensor([[True, True], [False, False]]),
            torch.tensor([[0.2, 0.7], [0.0, 0.0]]),
        ),
        (
            torch.tensor([0.1, -0.9, 0.1, -0.2]),
            torch.tensor(data=[False, True, False, False]),
            torch.tensor([0.0, -0.9, 0.0, 0.0]),
        ),
        (
            torch.tensor([[0.1, -0.9, 0.1, -0.2]]),
            torch.tensor(data=[False, True, False, False]),
            torch.tensor([0.0, -0.9, 0.0, 0.0]),
        ),
    ],
)
def test_mask_constraint(x, mask, expected):
    constraint = MaskConstraint(mask=mask)
    projected = constraint(x)
    assert_tensor_equal(projected, expected)


def test_quantization_constraint_invalid_mask():
    # test that passing a mask with shape different than x raises an error
    constraint = MaskConstraint(mask=torch.Tensor([1, 0]))
    with pytest.raises(ValueError):  # noqa: PT011
        constraint(torch.Tensor([[0]]))
