from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.nested import pad_along_batch, pad_along_seq, pad_sequences

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


#####################################
#     Tests for the side option     #
#####################################


def test_pad_along_batch_side_left() -> None:
    assert objects_are_equal(
        pad_along_batch(
            {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}, 1, side="left"
        ),
        {"a": torch.tensor([[0, 0], [0, 1], [2, 3]]), "b": torch.tensor([0, 4, 5])},
    )


def test_pad_along_seq_side_left() -> None:
    assert objects_are_equal(
        pad_along_seq([torch.tensor([[1, 2], [3, 4]])], 1, side="left"),
        [torch.tensor([[0, 1, 2], [0, 3, 4]])],
    )


def test_pad_along_batch_invalid_side() -> None:
    side: Any = "top"
    with pytest.raises(ValueError, match="side must be 'left' or 'right' but received 'top'"):
        pad_along_batch(torch.ones(2), 1, side=side)


###################################
#     Tests for pad_sequences     #
###################################


def test_pad_sequences_tensor() -> None:
    assert objects_are_equal(
        pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4])]),
        torch.tensor([[1, 2, 3], [4, 0, 0]]),
    )


def test_pad_sequences_dict() -> None:
    samples = [
        {"x": torch.tensor([1, 2, 3]), "mask": torch.tensor([True, True, True])},
        {"x": torch.tensor([4]), "mask": torch.tensor([True])},
    ]
    assert objects_are_equal(
        pad_sequences(samples),
        {
            "x": torch.tensor([[1, 2, 3], [4, 0, 0]]),
            "mask": torch.tensor([[True, True, True], [True, False, False]]),
        },
    )


def test_pad_sequences_list() -> None:
    samples: Any = [
        [torch.tensor([1, 2]), torch.tensor([3])],
        [torch.tensor([4]), torch.tensor([5, 6])],
    ]
    assert objects_are_equal(
        pad_sequences(samples),
        [torch.tensor([[1, 2], [4, 0]]), torch.tensor([[3, 0], [5, 6]])],
    )


def test_pad_sequences_tuple() -> None:
    samples = [(torch.tensor([1, 2]),), (torch.tensor([3]),)]
    assert objects_are_equal(pad_sequences(samples), (torch.tensor([[1, 2], [3, 0]]),))


def test_pad_sequences_nested_mix() -> None:
    samples: Any = [
        {"a": [torch.tensor([1, 2])], "b": {"c": torch.tensor([5])}},
        {"a": [torch.tensor([3])], "b": {"c": torch.tensor([6, 7])}},
    ]
    assert objects_are_equal(
        pad_sequences(samples),
        {"a": [torch.tensor([[1, 2], [3, 0]])], "b": {"c": torch.tensor([[5, 0], [6, 7]])}},
    )


def test_pad_sequences_trailing_dims() -> None:
    out = pad_sequences([{"a": torch.ones(3, 2)}, {"a": torch.ones(1, 2)}])
    assert objects_are_equal(
        out, {"a": torch.tensor([[[1.0, 1.0]] * 3, [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])}
    )


def test_pad_sequences_value() -> None:
    samples = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3])}]
    assert objects_are_equal(
        pad_sequences(samples, value=-1), {"a": torch.tensor([[1, 2], [3, -1]])}
    )


def test_pad_sequences_side_left() -> None:
    samples = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3])}]
    assert objects_are_equal(
        pad_sequences(samples, side="left"), {"a": torch.tensor([[1, 2], [0, 3]])}
    )


def test_pad_sequences_single_sample() -> None:
    assert objects_are_equal(
        pad_sequences([{"a": torch.tensor([1, 2])}]), {"a": torch.tensor([[1, 2]])}
    )


def test_pad_sequences_same_lengths() -> None:
    samples = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3, 4])}]
    assert objects_are_equal(pad_sequences(samples), {"a": torch.tensor([[1, 2], [3, 4]])})


def test_pad_sequences_dtype_is_the_one_of_the_first_sample() -> None:
    samples = [{"a": torch.tensor([1.5, 2.5])}, {"a": torch.tensor([1])}]
    assert objects_are_equal(pad_sequences(samples), {"a": torch.tensor([[1.5, 2.5], [1.0, 0.0]])})


def test_pad_sequences_key_order_of_the_first_sample() -> None:
    samples = [
        {"a": torch.tensor([1]), "b": torch.tensor([2])},
        {"b": torch.tensor([4]), "a": torch.tensor([3])},
    ]
    assert list(pad_sequences(samples)) == ["a", "b"]


def test_pad_sequences_does_not_modify_input() -> None:
    samples = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3])}]
    pad_sequences(samples)
    assert objects_are_equal(samples, [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3])}])


def test_pad_sequences_different_keys() -> None:
    with pytest.raises(ValueError, match="must have the same keys"):
        pad_sequences([{"a": torch.tensor([1])}, {"b": torch.tensor([1])}])


def test_pad_sequences_different_lengths_of_lists() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        pad_sequences(cast("Any", [[torch.tensor([1])], [torch.tensor([1]), torch.tensor([2])]]))


def test_pad_sequences_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty sequence of structures"):
        pad_sequences([])


def test_pad_sequences_different_trailing_shapes() -> None:
    with pytest.raises(ValueError, match="must have the same shape except for the first dimension"):
        pad_sequences([{"a": torch.ones(2, 3)}, {"a": torch.ones(2, 4)}])


def test_pad_sequences_scalar_leaf() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        pad_sequences([{"a": torch.tensor(1.0)}])


def test_pad_sequences_non_tensor_leaf() -> None:
    samples: Any = [{"a": "abc"}, {"a": "def"}]
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        pad_sequences(samples)


def test_pad_sequences_invalid_side() -> None:
    side: Any = "top"
    with pytest.raises(ValueError, match="side must be 'left' or 'right' but received 'top'"):
        pad_sequences([{"a": torch.tensor([1])}], side=side)


def test_pad_sequences_invalid_value() -> None:
    with pytest.raises(ValueError, match="cannot be represented"):
        pad_sequences([{"a": torch.tensor([1])}], value=1.5)
