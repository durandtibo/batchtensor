from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import pad_along_batch, pad_along_seq

#####################################
#     Tests for pad_along_batch     #
#####################################


def test_pad_along_batch_pad_size_0() -> None:
    tensor = torch.tensor([[0, 1], [2, 3], [4, 5]])
    assert objects_are_equal(pad_along_batch(tensor, pad_size=0), tensor)


def test_pad_along_batch_pad_size_2() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5]]), pad_size=2),
        torch.tensor([[0, 1], [2, 3], [4, 5], [0, 0], [0, 0]]),
    )


def test_pad_along_batch_value() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.tensor([[0, 1], [2, 3]]), pad_size=1, value=-1),
        torch.tensor([[0, 1], [2, 3], [-1, -1]]),
    )


def test_pad_along_batch_1d() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.tensor([1, 2, 3]), pad_size=2),
        torch.tensor([1, 2, 3, 0, 0]),
    )


def test_pad_along_batch_negative_pad_size() -> None:
    with pytest.raises(ValueError, match="pad_size must be a non-negative integer"):
        pad_along_batch(torch.tensor([1, 2, 3]), pad_size=-1)


###################################
#     Tests for pad_along_seq     #
###################################


def test_pad_along_seq_pad_size_0() -> None:
    tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert objects_are_equal(pad_along_seq(tensor, pad_size=0), tensor)


def test_pad_along_seq_pad_size_2() -> None:
    assert objects_are_equal(
        pad_along_seq(torch.tensor([[0, 1, 2], [3, 4, 5]]), pad_size=2),
        torch.tensor([[0, 1, 2, 0, 0], [3, 4, 5, 0, 0]]),
    )


def test_pad_along_seq_value() -> None:
    assert objects_are_equal(
        pad_along_seq(torch.tensor([[0, 1], [2, 3]]), pad_size=1, value=-1),
        torch.tensor([[0, 1, -1], [2, 3, -1]]),
    )


def test_pad_along_seq_3d() -> None:
    tensor = torch.zeros(2, 3, 4)
    out = pad_along_seq(tensor, pad_size=2)
    assert out.shape == (2, 5, 4)


def test_pad_along_seq_negative_pad_size() -> None:
    with pytest.raises(ValueError, match="pad_size must be a non-negative integer"):
        pad_along_seq(torch.tensor([[0, 1], [2, 3]]), pad_size=-1)
