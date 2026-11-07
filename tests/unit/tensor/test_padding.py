from __future__ import annotations

from typing import Any

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import lengths_to_mask, pad_along_batch, pad_along_seq, pad_sequences
from batchtensor.tensor.padding import _check_fill_value

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


def test_pad_along_batch_3d() -> None:
    out = pad_along_batch(torch.ones(2, 3, 4), pad_size=2, value=5.0)
    assert out.shape == (4, 3, 4)
    assert out[:2].eq(1).all()
    assert out[2:].eq(5).all()


def test_pad_along_seq_3d_values() -> None:
    out = pad_along_seq(torch.ones(2, 3, 4), pad_size=2, value=5.0)
    assert out[:, :3].eq(1).all()
    assert out[:, 3:].eq(5).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
def test_pad_along_seq_dtype_preserved(dtype: torch.dtype) -> None:
    out = pad_along_seq(torch.ones(2, 3, dtype=dtype), pad_size=1)
    assert out.dtype == dtype


def test_pad_along_seq_does_not_modify_input() -> None:
    tensor = torch.tensor([[0, 1], [2, 3]])
    pad_along_seq(tensor, pad_size=2)
    assert objects_are_equal(tensor, torch.tensor([[0, 1], [2, 3]]))


def test_pad_along_batch_empty_batch() -> None:
    out = pad_along_batch(torch.zeros(0, 3), pad_size=2, value=1.0)
    assert objects_are_equal(out, torch.ones(2, 3))


def test_pad_along_batch_int_tensor_float_value() -> None:
    with pytest.raises(ValueError, match=r"value \(1.5\) cannot be represented with dtype"):
        pad_along_batch(torch.arange(3), 1, value=1.5)


def test_pad_along_seq_int_tensor_float_value() -> None:
    with pytest.raises(ValueError, match=r"value \(0.5\) cannot be represented with dtype"):
        pad_along_seq(torch.ones(2, 3, dtype=torch.long), 1, value=0.5)


def test_pad_along_batch_int_tensor_integral_float_value() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.arange(3), 1, value=7.0), torch.tensor([0, 1, 2, 7])
    )


def test_pad_along_batch_pad_size_0_checks_value() -> None:
    with pytest.raises(ValueError, match="cannot be represented"):
        pad_along_batch(torch.arange(3), 0, value=1.5)


#####################################
#     Tests for _check_fill_value   #
#####################################


@pytest.mark.parametrize("value", [0, 1, -3.5, 1e30, float("inf"), float("nan")])
def test_check_fill_value_float(value: float) -> None:
    assert _check_fill_value(value, torch.float32) is None


def test_check_fill_value_complex() -> None:
    assert _check_fill_value(1.5, torch.complex64) is None


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, True])
def test_check_fill_value_bool(value: float) -> None:
    assert _check_fill_value(value, torch.bool) is None


@pytest.mark.parametrize("value", [2, -1, 0.5])
def test_check_fill_value_bool_invalid(value: float) -> None:
    with pytest.raises(ValueError, match=r"cannot be represented with dtype torch.bool"):
        _check_fill_value(value, torch.bool)


@pytest.mark.parametrize("value", [0, -128, 127, 3.0])
def test_check_fill_value_int8(value: float) -> None:
    assert _check_fill_value(value, torch.int8) is None


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 1.5])
def test_check_fill_value_int_not_integral(value: float) -> None:
    with pytest.raises(ValueError, match=r"cannot be represented with dtype torch.int64"):
        _check_fill_value(value, torch.int64)


@pytest.mark.parametrize("value", [128, -129])
def test_check_fill_value_int8_out_of_range(value: float) -> None:
    with pytest.raises(ValueError, match=r"out of range for dtype torch.int8 \[-128, 127\]"):
        _check_fill_value(value, torch.int8)


#####################################
#     Tests for the side option     #
#####################################


def test_pad_along_batch_side_left() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.tensor([[0, 1], [2, 3]]), 2, side="left"),
        torch.tensor([[0, 0], [0, 0], [0, 1], [2, 3]]),
    )


def test_pad_along_batch_side_right() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.tensor([[0, 1], [2, 3]]), 1, side="right"),
        torch.tensor([[0, 1], [2, 3], [0, 0]]),
    )


def test_pad_along_batch_side_left_value() -> None:
    assert objects_are_equal(
        pad_along_batch(torch.tensor([1.0, 2.0]), 1, value=-1.0, side="left"),
        torch.tensor([-1.0, 1.0, 2.0]),
    )


def test_pad_along_seq_side_left() -> None:
    assert objects_are_equal(
        pad_along_seq(torch.tensor([[1, 2], [3, 4]]), 2, side="left"),
        torch.tensor([[0, 0, 1, 2], [0, 0, 3, 4]]),
    )


def test_pad_along_seq_side_left_3d() -> None:
    assert objects_are_equal(
        pad_along_seq(torch.ones(2, 2, 3), 1, side="left")[:, 0],
        torch.zeros(2, 3),
    )


def test_pad_along_seq_side_left_pad_size_0() -> None:
    assert objects_are_equal(
        pad_along_seq(torch.tensor([[1, 2]]), 0, side="left"), torch.tensor([[1, 2]])
    )


@pytest.mark.parametrize("fn", [pad_along_batch, pad_along_seq])
def test_pad_invalid_side(fn: Any) -> None:
    with pytest.raises(ValueError, match="side must be 'left' or 'right' but received 'top'"):
        fn(torch.ones(2, 2), 1, side="top")


def test_pad_invalid_side_pad_size_0() -> None:
    side: Any = "top"
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        pad_along_batch(torch.ones(2), 0, side=side)


###################################
#     Tests for pad_sequences     #
###################################


def test_pad_sequences_ragged() -> None:
    assert objects_are_equal(
        pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4]), torch.tensor([5, 6])]),
        torch.tensor([[1, 2, 3], [4, 0, 0], [5, 6, 0]]),
    )


def test_pad_sequences_side_left() -> None:
    assert objects_are_equal(
        pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4])], side="left"),
        torch.tensor([[1, 2, 3], [0, 0, 4]]),
    )


def test_pad_sequences_value() -> None:
    assert objects_are_equal(
        pad_sequences([torch.tensor([1, 2]), torch.tensor([3])], value=-1),
        torch.tensor([[1, 2], [3, -1]]),
    )


def test_pad_sequences_equal_lengths() -> None:
    assert objects_are_equal(
        pad_sequences([torch.tensor([1, 2]), torch.tensor([3, 4])]),
        torch.tensor([[1, 2], [3, 4]]),
    )


def test_pad_sequences_single() -> None:
    assert objects_are_equal(pad_sequences([torch.tensor([1, 2])]), torch.tensor([[1, 2]]))


def test_pad_sequences_tuple_input() -> None:
    assert objects_are_equal(pad_sequences((torch.tensor([1]),)), torch.tensor([[1]]))


def test_pad_sequences_empty_sequences() -> None:
    assert objects_are_equal(
        pad_sequences([torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)]),
        torch.zeros(2, 0, dtype=torch.long),
    )


def test_pad_sequences_3d() -> None:
    out = pad_sequences([torch.ones(3, 2), torch.ones(1, 2)])
    assert objects_are_equal(out[1], torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def test_pad_sequences_3d_shape() -> None:
    assert pad_sequences([torch.ones(3, 2, 4), torch.ones(1, 2, 4)]).shape == (2, 3, 2, 4)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]
)
def test_pad_sequences_dtype_preserved(dtype: torch.dtype) -> None:
    assert pad_sequences([torch.ones(2, dtype=dtype), torch.ones(1, dtype=dtype)]).dtype == dtype


def test_pad_sequences_matches_lengths_to_mask() -> None:
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([4])]
    out = pad_sequences(sequences, value=-1)
    assert objects_are_equal(out != -1, lengths_to_mask(torch.tensor([3, 1])))


def test_pad_sequences_does_not_modify_input() -> None:
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([4])]
    pad_sequences(sequences)
    assert objects_are_equal(sequences, [torch.tensor([1, 2, 3]), torch.tensor([4])])


def test_pad_sequences_autograd() -> None:
    tensors = [torch.ones(2, requires_grad=True), torch.ones(1, requires_grad=True)]
    pad_sequences(tensors).sum().backward()
    assert objects_are_equal(tensors[0].grad, torch.ones(2))


def test_pad_sequences_empty() -> None:
    with pytest.raises(ValueError, match="at least one tensor"):
        pad_sequences([])


def test_pad_sequences_scalar() -> None:
    with pytest.raises(ValueError, match="at least one dimension but received a scalar"):
        pad_sequences([torch.tensor(1.0)])


def test_pad_sequences_later_scalar() -> None:
    with pytest.raises(ValueError, match=r"tensor 1 has shape \(\)"):
        pad_sequences([torch.ones(2), torch.tensor(1.0)])


def test_pad_sequences_trailing_shape_mismatch() -> None:
    with pytest.raises(
        ValueError, match=r"tensor 0 has shape \(2, 3\) and tensor 1 has shape \(1, 4\)"
    ):
        pad_sequences([torch.ones(2, 3), torch.ones(1, 4)])


def test_pad_sequences_value_not_representable() -> None:
    with pytest.raises(ValueError, match=r"value \(1.5\) cannot be represented"):
        pad_sequences([torch.arange(3)], value=1.5)


def test_pad_sequences_invalid_side() -> None:
    side: Any = "top"
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        pad_sequences([torch.ones(2)], side=side)
