from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import (
    cat_along_batch,
    cat_along_seq,
    repeat_along_seq,
    stack_along_batch,
    stack_along_seq,
)

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


######################################
#     Tests for repeat_along_seq     #
######################################


def test_repeat_along_seq_repeats_0() -> None:
    assert objects_are_equal(
        repeat_along_seq(
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), repeats=0
        ),
        torch.zeros(2, 0),
    )


def test_repeat_along_seq_repeats_1() -> None:
    assert objects_are_equal(
        repeat_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), repeats=1),
        torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    )


def test_repeat_along_seq_repeats_2() -> None:
    assert objects_are_equal(
        repeat_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), repeats=2),
        torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
    )


def test_repeat_along_seq_repeats_3d() -> None:
    assert objects_are_equal(
        repeat_along_seq(
            torch.tensor(
                [
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ]
            ),
            repeats=2,
        ),
        torch.tensor(
            [
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                ],
                [
                    [10, 11],
                    [12, 13],
                    [14, 15],
                    [16, 17],
                    [18, 19],
                    [10, 11],
                    [12, 13],
                    [14, 15],
                    [16, 17],
                    [18, 19],
                ],
            ]
        ),
    )


#######################################
#     Tests for stack_along_batch     #
#######################################


def test_stack_along_batch() -> None:
    assert objects_are_equal(
        stack_along_batch([torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]),
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )


def test_stack_along_batch_tuple() -> None:
    assert objects_are_equal(
        stack_along_batch((torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5]))),
        torch.tensor([[0, 1], [2, 3], [4, 5]]),
    )


#####################################
#     Tests for stack_along_seq     #
#####################################


def test_stack_along_seq() -> None:
    assert objects_are_equal(
        stack_along_seq([torch.tensor([0, 1]), torch.tensor([2, 3])]),
        torch.tensor([[0, 2], [1, 3]]),
    )


def test_stack_along_seq_tuple() -> None:
    assert objects_are_equal(
        stack_along_seq((torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5]))),
        torch.tensor([[0, 2, 4], [1, 3, 5]]),
    )
