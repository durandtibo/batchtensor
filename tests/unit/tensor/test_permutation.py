from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.tensor import permute_along_batch, permute_along_seq

INDEX_DTYPES = [torch.int, torch.long]

#########################################
#     Tests for permute_along_batch     #
#########################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        permute_along_batch(
            torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0], dtype=dtype)
        ),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


def test_permute_along_batch_incorrect_shape() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"permutation shape \(.*\) is not compatible with tensor shape \(.*\)",
    ):
        permute_along_batch(torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0, 2, 0]))


#######################################
#     Tests for permute_along_seq     #
#######################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        permute_along_seq(torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0], dtype=dtype)),
        torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


def test_permute_along_seq_incorrect_shape() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"permutation shape \(.*\) is not compatible with tensor shape \(.*\)",
    ):
        permute_along_seq(torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0, 2, 0]))
