from __future__ import annotations

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
