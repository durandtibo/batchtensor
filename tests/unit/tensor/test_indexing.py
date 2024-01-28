from __future__ import annotations

import torch
from coola import objects_are_equal

from batchtensor.tensor import index_select_along_batch, index_select_along_seq

##############################################
#     Tests for index_select_along_batch     #
##############################################


def test_index_select_along_batch_2() -> None:
    assert objects_are_equal(
        index_select_along_batch(torch.arange(10).view(5, 2), torch.tensor([2, 4])),
        torch.tensor([[4, 5], [8, 9]]),
    )


def test_index_select_along_batch_5() -> None:
    assert objects_are_equal(
        index_select_along_batch(torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0])),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


def test_index_select_along_batch_7() -> None:
    assert objects_are_equal(
        index_select_along_batch(torch.arange(10).view(5, 2), torch.tensor([4, 3, 2, 1, 0, 2, 0])),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1], [4, 5], [0, 1]]),
    )


############################################
#     Tests for index_select_along_seq     #
############################################


def test_index_select_along_seq_2() -> None:
    assert objects_are_equal(
        index_select_along_seq(torch.arange(10).view(2, 5), torch.tensor([2, 4])),
        torch.tensor([[2, 4], [7, 9]]),
    )


def test_index_select_along_seq_5() -> None:
    assert objects_are_equal(
        index_select_along_seq(torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0])),
        torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


def test_index_select_along_seq_7() -> None:
    assert objects_are_equal(
        index_select_along_seq(torch.arange(10).view(2, 5), torch.tensor([4, 3, 2, 1, 0, 2, 0])),
        torch.tensor([[4, 3, 2, 1, 0, 2, 0], [9, 8, 7, 6, 5, 7, 5]]),
    )
