from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola import objects_are_equal

from batchtensor.nested import cat_along_batch, cat_along_seq

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

#####################################
#     Tests for cat_along_batch     #
#####################################


@pytest.mark.parametrize(
    "data",
    [
        [
            {"a": torch.tensor([[0, 1, 2], [4, 5, 6]]), "b": torch.tensor([[7], [8]])},
            {"a": torch.tensor([[10, 11, 12], [14, 15, 16]]), "b": torch.tensor([[17], [18]])},
        ],
        (
            {"a": torch.tensor([[0, 1, 2], [4, 5, 6]]), "b": torch.tensor([[7], [8]])},
            {"a": torch.tensor([[10, 11, 12], [14, 15, 16]]), "b": torch.tensor([[17], [18]])},
        ),
        [
            {"a": torch.tensor([[0, 1, 2], [4, 5, 6]]), "b": torch.tensor([[7], [8]])},
            {"a": torch.tensor([[10, 11, 12]]), "b": torch.tensor([[17]])},
            {"a": torch.tensor([[14, 15, 16]]), "b": torch.tensor([[18]])},
        ],
    ],
)
def test_cat_along_batch(data: Sequence[dict[Hashable, torch.Tensor]]) -> None:
    assert objects_are_equal(
        cat_along_batch(data),
        {
            "a": torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [14, 15, 16]]),
            "b": torch.tensor([[7], [8], [17], [18]]),
        },
    )


def test_cat_along_batch_empty() -> None:
    assert objects_are_equal(cat_along_batch([]), {})


###################################
#     Tests for cat_along_seq     #
###################################


@pytest.mark.parametrize(
    "data",
    [
        [
            {"a": torch.tensor([[0, 1, 2], [4, 5, 6]]), "b": torch.tensor([[7], [8]])},
            {
                "a": torch.tensor([[10, 11, 12], [13, 14, 15]]),
                "b": torch.tensor([[17, 18], [18, 19]]),
            },
        ],
        (
            {"a": torch.tensor([[0, 1, 2], [4, 5, 6]]), "b": torch.tensor([[7], [8]])},
            {
                "a": torch.tensor([[10, 11, 12], [13, 14, 15]]),
                "b": torch.tensor([[17, 18], [18, 19]]),
            },
        ),
        [
            {"a": torch.tensor([[0, 1, 2], [4, 5, 6]]), "b": torch.tensor([[7], [8]])},
            {"a": torch.tensor([[10, 11], [13, 14]]), "b": torch.tensor([[17], [18]])},
            {"a": torch.tensor([[12], [15]]), "b": torch.tensor([[18], [19]])},
        ],
    ],
)
def test_cat_along_seq(data: Sequence[dict[Hashable, torch.Tensor]]) -> None:
    assert objects_are_equal(
        cat_along_seq(data),
        {
            "a": torch.tensor([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
            "b": torch.tensor([[7, 17, 18], [8, 18, 19]]),
        },
    )


def test_cat_along_seq_empty() -> None:
    assert objects_are_equal(cat_along_seq([]), {})
