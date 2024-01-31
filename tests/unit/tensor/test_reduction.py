from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.tensor import sum_along_batch, sum_along_seq

DTYPES = (torch.float, torch.long)

#####################################
#     Tests for sum_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([20, 25], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[20, 25]], dtype=dtype),
    )


###################################
#     Tests for sum_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([10, 35], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[10], [35]], dtype=dtype),
    )
