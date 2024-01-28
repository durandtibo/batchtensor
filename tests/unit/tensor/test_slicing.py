from __future__ import annotations

import torch
from coola import objects_are_equal

from batchtensor.tensor import slice_along_batch, slice_along_seq

#######################################
#     Tests for slice_along_batch     #
#######################################


def test_slice_along_batch() -> None:
    assert objects_are_equal(
        slice_along_batch(torch.arange(10).view(5, 2)),
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
    )


def test_slice_along_batch_start_2() -> None:
    assert objects_are_equal(
        slice_along_batch(torch.arange(10).view(5, 2), start=2),
        torch.tensor([[4, 5], [6, 7], [8, 9]]),
    )


def test_slice_along_batch_stop_3() -> None:
    assert objects_are_equal(
        slice_along_batch(torch.arange(10).view(5, 2), stop=3),
        torch.tensor([[0, 1], [2, 3], [4, 5]]),
    )


def test_slice_along_batch_stop_100() -> None:
    assert objects_are_equal(
        slice_along_batch(torch.arange(10).view(5, 2), stop=100),
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
    )


def test_slice_along_batch_step_2() -> None:
    assert objects_are_equal(
        slice_along_batch(torch.arange(10).view(5, 2), step=2),
        torch.tensor([[0, 1], [4, 5], [8, 9]]),
    )


def test_slice_along_batch_start_1_stop_4_step_2() -> None:
    assert objects_are_equal(
        slice_along_batch(torch.arange(10).view(5, 2), start=1, stop=4, step=2),
        torch.tensor([[2, 3], [6, 7]]),
    )


def test_slice_along_seq() -> None:
    assert objects_are_equal(
        slice_along_seq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])),
        torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
    )


def test_slice_along_seq_start_2() -> None:
    assert objects_are_equal(
        slice_along_seq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), start=2),
        torch.tensor([[2, 3, 4], [7, 6, 5]]),
    )


def test_slice_along_seq_stop_3() -> None:
    assert objects_are_equal(
        slice_along_seq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), stop=3),
        torch.tensor([[0, 1, 2], [9, 8, 7]]),
    )


def test_example_batch_slice_along_seq_stop_100() -> None:
    assert objects_are_equal(
        slice_along_seq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), stop=100),
        torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
    )


def test_slice_along_seq_step_2() -> None:
    assert objects_are_equal(
        slice_along_seq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), step=2),
        torch.tensor([[0, 2, 4], [9, 7, 5]]),
    )
