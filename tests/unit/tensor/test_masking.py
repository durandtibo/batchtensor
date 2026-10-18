from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import lengths_to_mask

######################################
#     Tests for lengths_to_mask     #
######################################


def test_lengths_to_mask_default_max_len() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1, 2])),
        torch.tensor(
            [[True, True, True], [True, False, False], [True, True, False]],
        ),
    )


def test_lengths_to_mask_explicit_max_len() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1, 2]), max_len=4),
        torch.tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, False],
            ],
        ),
    )


def test_lengths_to_mask_all_zero() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([0, 0])),
        torch.zeros(2, 0, dtype=torch.bool),
    )


def test_lengths_to_mask_empty() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([], dtype=torch.long)),
        torch.zeros(0, 0, dtype=torch.bool),
    )


def test_lengths_to_mask_not_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        lengths_to_mask(torch.tensor([[1, 2]]))


def test_lengths_to_mask_max_len_smaller_than_lengths() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1, 2]), max_len=2),
        torch.tensor([[True, True], [True, False], [True, True]]),
    )


def test_lengths_to_mask_max_len_zero() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1]), max_len=0),
        torch.zeros(2, 0, dtype=torch.bool),
    )


def test_lengths_to_mask_all_zero_with_max_len() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([0, 0]), max_len=3),
        torch.zeros(2, 3, dtype=torch.bool),
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_lengths_to_mask_dtype(dtype: torch.dtype) -> None:
    out = lengths_to_mask(torch.tensor([2, 1], dtype=dtype))
    assert out.dtype == torch.bool
    assert out.tolist() == [[True, True], [True, False]]


def test_lengths_to_mask_matches_pad_along_seq_layout() -> None:
    lengths = torch.tensor([2, 3])
    mask = lengths_to_mask(lengths, max_len=5)
    assert mask.sum(dim=1).tolist() == lengths.tolist()
