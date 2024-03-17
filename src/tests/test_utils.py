import pytest
import torch
from secmlt.utils.tensor_utils import atleast_kd


@pytest.mark.parametrize(
    "input_tensor, desired_dims, expected_shape",
    [
        (torch.tensor([1, 2, 3]), 2, (3, 1)),
        (torch.tensor([[1, 2], [3, 4]]), 3, (2, 2, 1)),
        (torch.tensor([[[1], [2]], [[3], [4]]]), 4, (2, 2, 1, 1)),
    ],
)
def test_atleast_kd(input_tensor, desired_dims, expected_shape):
    output_tensor = atleast_kd(input_tensor, desired_dims)
    assert output_tensor.shape == expected_shape


def test_atleast_kd_raises_error():
    x = torch.tensor([[1, 2], [3, 4]])
    msg = "The number of desired dimensions should be > x.dim()"
    with pytest.raises(ValueError, match=msg):
        atleast_kd(x, 1)
