import pytest
import torch
from coola import objects_are_equal

from batchtensor.tensor import cumsum_along_batch, cumsum_along_seq

DTYPES = [torch.float, torch.double, torch.long]

########################################
#     Tests for cumsum_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_cumsum_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        cumsum_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]], dtype=dtype),
        show_difference=True,
    )


######################################
#     Tests for cumsum_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_cumsum_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        cumsum_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]], dtype=dtype),
    )
