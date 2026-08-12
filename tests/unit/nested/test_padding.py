from __future__ import annotations

import torch
from coola.equality import objects_are_equal

from batchtensor.nested import pad_along_batch, pad_along_seq

#####################################
#     Tests for pad_along_batch     #
#####################################


def test_pad_along_batch() -> None:
    assert objects_are_equal(
        pad_along_batch(
            {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}, pad_size=1
        ),
        {"a": torch.tensor([[0, 1], [2, 3], [0, 0]]), "b": torch.tensor([4, 5, 0])},
    )


def test_pad_along_batch_value() -> None:
    assert objects_are_equal(
        pad_along_batch({"a": torch.tensor([0, 1, 2])}, pad_size=2, value=-1),
        {"a": torch.tensor([0, 1, 2, -1, -1])},
    )


def test_pad_along_batch_nested() -> None:
    assert objects_are_equal(
        pad_along_batch({"a": torch.tensor([0, 1]), "b": {"c": torch.tensor([2, 3])}}, pad_size=1),
        {"a": torch.tensor([0, 1, 0]), "b": {"c": torch.tensor([2, 3, 0])}},
    )


###################################
#     Tests for pad_along_seq     #
###################################


def test_pad_along_seq() -> None:
    assert objects_are_equal(
        pad_along_seq(
            {"a": torch.tensor([[0, 1, 2], [3, 4, 5]]), "b": torch.tensor([[6, 7, 8]])},
            pad_size=1,
        ),
        {"a": torch.tensor([[0, 1, 2, 0], [3, 4, 5, 0]]), "b": torch.tensor([[6, 7, 8, 0]])},
    )


def test_pad_along_seq_value() -> None:
    assert objects_are_equal(
        pad_along_seq({"a": torch.tensor([[0, 1], [2, 3]])}, pad_size=1, value=-1),
        {"a": torch.tensor([[0, 1, -1], [2, 3, -1]])},
    )
