from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.tensor import cat_along_batch, cat_along_seq

#####################################
#     Tests for cat_along_batch     #
#####################################


@pytest.mark.parametrize(
    "tensors",
    [
        [torch.tensor([[0, 1, 2], [4, 5, 6]]), torch.tensor([[10, 11, 12], [13, 14, 15]])],
        (torch.tensor([[0, 1, 2], [4, 5, 6]]), torch.tensor([[10, 11, 12], [13, 14, 15]])),
        [
            torch.tensor([[0, 1, 2], [4, 5, 6]]),
            torch.tensor([[10, 11, 12]]),
            torch.tensor([[13, 14, 15]]),
        ],
        [
            torch.tensor([[0, 1, 2], [4, 5, 6]]),
            torch.ones(0, 3, dtype=torch.long),
            torch.tensor([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_cat_along_batch(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:
    assert objects_are_equal(
        cat_along_batch(tensors),
        torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


###################################
#     Tests for cat_along_seq     #
###################################


@pytest.mark.parametrize(
    "tensors",
    [
        [torch.tensor([[0, 1, 2], [4, 5, 6]]), torch.tensor([[10, 11, 12], [13, 14, 15]])],
        (torch.tensor([[0, 1, 2], [4, 5, 6]]), torch.tensor([[10, 11, 12], [13, 14, 15]])),
        [
            torch.tensor([[0, 1, 2], [4, 5, 6]]),
            torch.tensor([[10, 11], [13, 14]]),
            torch.tensor([[12], [15]]),
        ],
        [
            torch.tensor([[0, 1, 2], [4, 5, 6]]),
            torch.ones(2, 0, dtype=torch.long),
            torch.tensor([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_cat_along_seq(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:
    assert objects_are_equal(
        cat_along_seq(tensors),
        torch.tensor([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
    )
