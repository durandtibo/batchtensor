from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.tensor import index_select_along_batch, index_select_along_seq

INDEX_DTYPES = [torch.int, torch.long]

##############################################
#     Tests for index_select_along_batch     #
##############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_2(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_batch(torch.arange(10).view(5, 2), torch.tensor([2, 4], dtype=dtype)),
        torch.tensor([[4, 5], [8, 9]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_5(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0], dtype=dtype)
        ),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_7(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0, 2, 0], dtype=dtype)
        ),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1], [4, 5], [0, 1]]),
    )


############################################
#     Tests for index_select_along_seq     #
############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_2(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_seq(torch.arange(10).view(2, 5), torch.tensor([2, 4], dtype=dtype)),
        torch.tensor([[2, 4], [7, 9]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_5(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0], dtype=dtype)
        ),
        torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_7(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0, 2, 0], dtype=dtype)
        ),
        torch.tensor([[4, 3, 2, 1, 0, 2, 0], [9, 8, 7, 6, 5, 7, 5]]),
    )
