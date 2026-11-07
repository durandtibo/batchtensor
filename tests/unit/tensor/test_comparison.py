from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import (
    argsort_along_batch,
    argsort_along_seq,
    sort_along_batch,
    sort_along_seq,
    topk_along_batch,
    topk_along_seq,
)
from tests.conftest import torch_greater_equal_1_13

#########################################
#     Tests for argsort_along_batch     #
#########################################


def test_argsort_along_batch_descending_false() -> None:
    assert objects_are_equal(
        argsort_along_batch(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
        torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]),
    )


def test_argsort_along_batch_descending_true() -> None:
    assert objects_are_equal(
        argsort_along_batch(
            torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), descending=True
        ),
        torch.tensor([[3, 0], [0, 4], [4, 1], [2, 3], [1, 2]]),
    )


@torch_greater_equal_1_13
def test_argsort_along_batch_stable_true() -> None:
    assert objects_are_equal(
        argsort_along_batch(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), stable=True),
        torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]),
    )


#######################################
#     Tests for argsort_along_seq     #
#######################################


def test_argsort_along_seq_descending_false() -> None:
    assert objects_are_equal(
        argsort_along_seq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]),
    )


def test_argsort_along_seq_descending_true() -> None:
    assert objects_are_equal(
        argsort_along_seq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), descending=True),
        torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]]),
    )


@torch_greater_equal_1_13
def test_argsort_along_seq_stable_true() -> None:
    assert objects_are_equal(
        argsort_along_seq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), stable=True),
        torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]),
    )


######################################
#     Tests for sort_along_batch     #
######################################


def test_sort_along_batch_descending_false() -> None:
    assert objects_are_equal(
        sort_along_batch(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
        torch.return_types.sort(
            (
                torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]),
                torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]),
            )
        ),
    )


def test_sort_along_batch_descending_true() -> None:
    assert objects_are_equal(
        sort_along_batch(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), descending=True),
        torch.return_types.sort(
            (
                torch.tensor([[5, 9], [4, 8], [3, 7], [2, 6], [1, 5]]),
                torch.tensor([[3, 0], [0, 4], [4, 1], [2, 3], [1, 2]]),
            )
        ),
    )


@torch_greater_equal_1_13
def test_sort_along_batch_stable_true() -> None:
    assert objects_are_equal(
        sort_along_batch(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [2, 9]]), stable=True),
        torch.return_types.sort(
            (
                torch.tensor([[1, 5], [2, 6], [2, 7], [4, 9], [5, 9]]),
                torch.tensor([[1, 2], [2, 3], [4, 1], [0, 0], [3, 4]]),
            )
        ),
    )


####################################
#     Tests for sort_along_seq     #
####################################


def test_sort_along_seq_descending_false() -> None:
    assert objects_are_equal(
        sort_along_seq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        torch.return_types.sort(
            (
                torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]),
                torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]),
            )
        ),
    )


def test_sort_along_seq_descending_true() -> None:
    assert objects_are_equal(
        sort_along_seq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), descending=True),
        torch.return_types.sort(
            (
                torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]]),
                torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]]),
            )
        ),
    )


@torch_greater_equal_1_13
def test_sort_along_seq_stable_true() -> None:
    assert objects_are_equal(
        sort_along_seq(torch.tensor([[4, 1, 3, 5, 3], [9, 7, 5, 6, 9]]), stable=True),
        torch.return_types.sort(
            (
                torch.tensor([[1, 3, 3, 4, 5], [5, 6, 7, 9, 9]]),
                torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 0, 4]]),
            )
        ),
    )


#########################################################
#     Tests for topk_along_batch and topk_along_seq     #
#########################################################


def test_topk_along_batch_k_2_values() -> None:
    assert objects_are_equal(
        topk_along_batch(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2).values,
        torch.tensor([[8, 9, 5, 6], [4, 7, 3, 2]]),
    )


def test_topk_along_batch_k_2_indices() -> None:
    assert objects_are_equal(
        topk_along_batch(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2).indices,
        torch.tensor([[2, 1, 1, 1], [1, 2, 0, 0]]),
    )


def test_topk_along_batch_k_2_largest_false_values() -> None:
    assert objects_are_equal(
        topk_along_batch(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2, largest=False
        ).values,
        torch.tensor([[0, 1, 2, 1], [4, 7, 3, 2]]),
    )


def test_topk_along_batch_k_2_largest_false_indices() -> None:
    assert objects_are_equal(
        topk_along_batch(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2, largest=False
        ).indices,
        torch.tensor([[0, 0, 2, 2], [1, 2, 0, 0]]),
    )


def test_topk_along_batch_k_3_sort_output_false_values() -> None:
    assert objects_are_equal(
        topk_along_batch(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 3, sort_output=False
        ).values,
        torch.tensor([[8, 9, 5, 6], [4, 7, 3, 2], [0, 1, 2, 1]]),
    )


def test_topk_along_batch_k_3_sort_output_false_indices() -> None:
    assert objects_are_equal(
        topk_along_batch(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 3, sort_output=False
        ).indices,
        torch.tensor([[2, 1, 1, 1], [1, 2, 0, 0], [0, 0, 2, 2]]),
    )


def test_topk_along_batch_returns_named_tuple() -> None:
    assert isinstance(
        topk_along_batch(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2),
        torch.return_types.topk,
    )


def test_topk_along_batch_does_not_modify_input() -> None:
    tensor = torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])
    topk_along_batch(tensor, 2)
    assert objects_are_equal(tensor, torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]))


def test_topk_along_seq_k_2_values() -> None:
    assert objects_are_equal(
        topk_along_seq(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2).values,
        torch.tensor([[3, 2], [9, 6], [8, 7]]),
    )


def test_topk_along_seq_k_2_indices() -> None:
    assert objects_are_equal(
        topk_along_seq(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2).indices,
        torch.tensor([[2, 3], [1, 3], [0, 1]]),
    )


def test_topk_along_seq_k_2_largest_false_values() -> None:
    assert objects_are_equal(
        topk_along_seq(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2, largest=False
        ).values,
        torch.tensor([[0, 1], [4, 5], [1, 2]]),
    )


def test_topk_along_seq_k_2_largest_false_indices() -> None:
    assert objects_are_equal(
        topk_along_seq(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2, largest=False
        ).indices,
        torch.tensor([[0, 1], [0, 2], [3, 2]]),
    )


def test_topk_along_seq_k_3_sort_output_false_values() -> None:
    assert objects_are_equal(
        topk_along_seq(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 3, sort_output=False
        ).values,
        torch.tensor([[3, 2, 1], [9, 6, 5], [8, 7, 2]]),
    )


def test_topk_along_seq_k_3_sort_output_false_indices() -> None:
    assert objects_are_equal(
        topk_along_seq(
            torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 3, sort_output=False
        ).indices,
        torch.tensor([[2, 3, 1], [1, 3, 2], [0, 1, 2]]),
    )


def test_topk_along_seq_returns_named_tuple() -> None:
    assert isinstance(
        topk_along_seq(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 2),
        torch.return_types.topk,
    )


def test_topk_along_seq_does_not_modify_input() -> None:
    tensor = torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])
    topk_along_seq(tensor, 2)
    assert objects_are_equal(tensor, torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]))


def test_topk_along_seq_1d_tensor() -> None:
    with pytest.raises(IndexError, match="Dimension out of range"):
        topk_along_seq(torch.tensor([0, 4, 8]), 2)
