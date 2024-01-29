from __future__ import annotations

import torch
from coola import objects_are_equal

from batchtensor.nested import chunk_along_batch, chunk_along_seq

INDEX_DTYPES = [torch.int, torch.long]

#######################################
#     Tests for chunk_along_batch     #
#######################################


def test_chunk_along_batch_chunks_3() -> None:
    assert objects_are_equal(
        chunk_along_batch(
            {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}, chunks=3
        ),
        (
            {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 3])},
            {"a": torch.tensor([[4, 5], [6, 7]]), "b": torch.tensor([2, 1])},
            {"a": torch.tensor([[8, 9]]), "b": torch.tensor([0])},
        ),
    )


def test_chunk_along_batch_chunks_5() -> None:
    assert objects_are_equal(
        chunk_along_batch(
            {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}, chunks=5
        ),
        (
            {"a": torch.tensor([[0, 1]]), "b": torch.tensor([4])},
            {"a": torch.tensor([[2, 3]]), "b": torch.tensor([3])},
            {"a": torch.tensor([[4, 5]]), "b": torch.tensor([2])},
            {"a": torch.tensor([[6, 7]]), "b": torch.tensor([1])},
            {"a": torch.tensor([[8, 9]]), "b": torch.tensor([0])},
        ),
    )


#####################################
#     Tests for chunk_along_seq     #
#####################################


def test_chunk_along_seq_chunks_3() -> None:
    assert objects_are_equal(
        chunk_along_seq(
            {"a": torch.arange(10).view(2, 5), "b": torch.tensor([[4, 3, 2, 1, 0]])}, chunks=3
        ),
        (
            {"a": torch.tensor([[0, 1], [5, 6]]), "b": torch.tensor([[4, 3]])},
            {"a": torch.tensor([[2, 3], [7, 8]]), "b": torch.tensor([[2, 1]])},
            {"a": torch.tensor([[4], [9]]), "b": torch.tensor([[0]])},
        ),
    )


def test_chunk_along_seq_chunks_5() -> None:
    assert objects_are_equal(
        chunk_along_seq(
            {"a": torch.arange(10).view(2, 5), "b": torch.tensor([[4, 3, 2, 1, 0]])}, chunks=5
        ),
        (
            {"a": torch.tensor([[0], [5]]), "b": torch.tensor([[4]])},
            {"a": torch.tensor([[1], [6]]), "b": torch.tensor([[3]])},
            {"a": torch.tensor([[2], [7]]), "b": torch.tensor([[2]])},
            {"a": torch.tensor([[3], [8]]), "b": torch.tensor([[1]])},
            {"a": torch.tensor([[4], [9]]), "b": torch.tensor([[0]])},
        ),
    )
