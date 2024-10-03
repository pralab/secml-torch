import pytest
import torch
from secmlt.optimization.constraints import (
    ClipConstraint,
    L0Constraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
    MaskConstraint,
    QuantizationConstraint,
)


def assert_tensor_equal(actual, expected, msg="") -> None:
    assert torch.allclose(
        actual, expected
    ), f"Expected {expected}, but got {actual}. {msg}"


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
    assert torch.all(
        norms <= expected_norm
    ), f"{constraint_class.__name__} failed with norms {norms}"


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
    "x, levels, expected",
    [
        (
            torch.tensor([[0.2, 0.7], [0.4, 0.8]]),
            5,
            torch.tensor([[0.25, 0.75], [0.5, 0.75]]),
        ),
        (
            torch.tensor([[0.1, 0.9], [0.3, 0.6]]),
            3,
            torch.tensor([[0.0, 1.0], [0.5, 0.5]]),
        ),
    ],
)
def test_quantization_constraint(x, levels, expected):
    constraint = QuantizationConstraint(levels=levels)
    projected = constraint(x)
    assert_tensor_equal(projected, expected)


def test_quantization_constraint_invalid_levels():
    # test that passing a non-integer levels value raises an error
    with pytest.raises(ValueError):  # noqa: PT011
        QuantizationConstraint(levels=2.5)


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
