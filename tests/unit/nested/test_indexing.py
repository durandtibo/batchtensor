from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.nested import index_select_along_batch, index_select_along_seq

INDEX_DTYPES = [torch.int, torch.long]

##############################################
#     Tests for index_select_along_batch     #
##############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0], dtype=dtype)
        ),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            {"a": torch.arange(10).view(5, 2), "b": torch.tensor([[5], [4], [3], [2], [1]])},
            torch.tensor([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
            "b": torch.tensor([[1], [2], [3], [4], [5]]),
        },
    )


############################################
#     Tests for index_select_along_seq     #
############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0], dtype=dtype)
        ),
        torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            {"a": torch.arange(10).view(2, 5), "b": torch.tensor([[5, 4, 3, 2, 1]])},
            torch.tensor([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
            "b": torch.tensor([[1, 2, 3, 4, 5]]),
        },
    )
