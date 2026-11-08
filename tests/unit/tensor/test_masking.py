from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import (
    lengths_to_mask,
    masked_amax_along_seq,
    masked_amin_along_seq,
    masked_mean_along_seq,
    masked_sum_along_seq,
)
from batchtensor.tensor.masking import _fill_invalid, _highest, _lowest

if TYPE_CHECKING:
    from collections.abc import Callable

#####################################
#     Tests for lengths_to_mask     #
#####################################


def test_lengths_to_mask_default_max_len() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1, 2])),
        torch.tensor(
            [[True, True, True], [True, False, False], [True, True, False]],
        ),
    )


def test_lengths_to_mask_explicit_max_len() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1, 2]), max_len=4),
        torch.tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, False],
            ],
        ),
    )


def test_lengths_to_mask_all_zero() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([0, 0])),
        torch.zeros(2, 0, dtype=torch.bool),
    )


def test_lengths_to_mask_empty() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([], dtype=torch.long)),
        torch.zeros(0, 0, dtype=torch.bool),
    )


def test_lengths_to_mask_not_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        lengths_to_mask(torch.tensor([[1, 2]]))


def test_lengths_to_mask_max_len_smaller_than_lengths() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1, 2]), max_len=2),
        torch.tensor([[True, True], [True, False], [True, True]]),
    )


def test_lengths_to_mask_max_len_zero() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([3, 1]), max_len=0),
        torch.zeros(2, 0, dtype=torch.bool),
    )


def test_lengths_to_mask_all_zero_with_max_len() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([0, 0]), max_len=3),
        torch.zeros(2, 3, dtype=torch.bool),
    )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int32, torch.int64])
def test_lengths_to_mask_dtype(dtype: torch.dtype) -> None:
    out = lengths_to_mask(torch.tensor([2, 1], dtype=dtype))
    assert out.dtype == torch.bool
    assert out.tolist() == [[True, True], [True, False]]


def test_lengths_to_mask_matches_pad_along_seq_layout() -> None:
    lengths = torch.tensor([2, 3])
    mask = lengths_to_mask(lengths, max_len=5)
    assert mask.sum(dim=1).tolist() == lengths.tolist()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bool])
def test_lengths_to_mask_non_integer_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="must be an integer tensor but received dtype"):
        lengths_to_mask(torch.tensor([1, 0], dtype=dtype))


def test_lengths_to_mask_negative_length() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([-1, 2])),
        torch.tensor([[False, False], [True, True]]),
    )


def test_lengths_to_mask_all_negative_lengths() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([-3, -1])), torch.zeros(2, 0, dtype=torch.bool)
    )


def test_lengths_to_mask_max_len_smaller_than_length_truncates() -> None:
    assert objects_are_equal(
        lengths_to_mask(torch.tensor([5, 2]), max_len=3),
        torch.tensor([[True, True, True], [True, True, False]]),
    )


MASKED_FUNCTIONS = [
    masked_sum_along_seq,
    masked_mean_along_seq,
    masked_amax_along_seq,
    masked_amin_along_seq,
]
NAN = float("nan")


def make_tensor() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def make_mask() -> torch.Tensor:
    return torch.tensor([[True, True, False], [True, False, False]])


##########################################
#     Tests for masked_sum_along_seq     #
##########################################


def test_masked_sum_along_seq() -> None:
    assert objects_are_equal(
        masked_sum_along_seq(make_tensor(), make_mask()), torch.tensor([3.0, 4.0])
    )


def test_masked_sum_along_seq_all_true() -> None:
    assert objects_are_equal(
        masked_sum_along_seq(make_tensor(), torch.ones(2, 3, dtype=torch.bool)),
        torch.tensor([6.0, 15.0]),
    )


def test_masked_sum_along_seq_all_false_row() -> None:
    mask = torch.tensor([[False, False, False], [True, True, True]])
    assert objects_are_equal(masked_sum_along_seq(make_tensor(), mask), torch.tensor([0.0, 15.0]))


def test_masked_sum_along_seq_keepdim() -> None:
    assert objects_are_equal(
        masked_sum_along_seq(make_tensor(), make_mask(), keepdim=True),
        torch.tensor([[3.0], [4.0]]),
    )


def test_masked_sum_along_seq_int() -> None:
    assert objects_are_equal(
        masked_sum_along_seq(torch.tensor([[1, 2, 3], [4, 5, 6]]), make_mask()),
        torch.tensor([3, 4]),
    )


def test_masked_sum_along_seq_trailing_dims() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert objects_are_equal(
        masked_sum_along_seq(tensor, make_mask()), torch.tensor([[2.0, 4.0], [6.0, 7.0]])
    )


def test_masked_sum_along_seq_4d() -> None:
    out = masked_sum_along_seq(torch.ones(2, 3, 4, 5), make_mask())
    assert objects_are_equal(out, torch.tensor([[[2.0] * 5] * 4, [[1.0] * 5] * 4]))


def test_masked_sum_along_seq_matches_manual_form() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).sin()
    mask = make_mask()
    assert objects_are_equal(
        masked_sum_along_seq(tensor, mask), tensor.masked_fill(~mask[..., None], 0.0).sum(dim=1)
    )


###########################################
#     Tests for masked_mean_along_seq     #
###########################################


def test_masked_mean_along_seq() -> None:
    assert objects_are_equal(
        masked_mean_along_seq(make_tensor(), make_mask()), torch.tensor([1.5, 4.0])
    )


def test_masked_mean_along_seq_all_true() -> None:
    assert objects_are_equal(
        masked_mean_along_seq(make_tensor(), torch.ones(2, 3, dtype=torch.bool)),
        make_tensor().mean(dim=1),
    )


def test_masked_mean_along_seq_all_false_row_is_nan() -> None:
    mask = torch.tensor([[False, False, False], [True, True, True]])
    assert objects_are_equal(
        masked_mean_along_seq(make_tensor(), mask),
        torch.tensor([NAN, 5.0]),
        equal_nan=True,
    )


def test_masked_mean_along_seq_keepdim() -> None:
    assert objects_are_equal(
        masked_mean_along_seq(make_tensor(), make_mask(), keepdim=True),
        torch.tensor([[1.5], [4.0]]),
    )


def test_masked_mean_along_seq_trailing_dims() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert objects_are_equal(
        masked_mean_along_seq(tensor, make_mask()), torch.tensor([[1.0, 2.0], [6.0, 7.0]])
    )


def test_masked_mean_along_seq_keepdim_trailing_dims() -> None:
    out = masked_mean_along_seq(torch.ones(2, 3, 4), make_mask(), keepdim=True)
    assert objects_are_equal(out, torch.ones(2, 1, 4))


def test_masked_mean_along_seq_float64() -> None:
    out = masked_mean_along_seq(make_tensor().double(), make_mask())
    assert objects_are_equal(out, torch.tensor([1.5, 4.0], dtype=torch.float64))


def test_masked_mean_along_seq_int_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"requires a floating point or complex tensor but received dtype torch.int64",
    ):
        masked_mean_along_seq(torch.tensor([[1, 2, 3], [4, 5, 6]]), make_mask())


def test_masked_mean_along_seq_matches_manual_form() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).sin()
    mask = make_mask()
    expected = tensor.masked_fill(~mask[..., None], 0.0).sum(dim=1) / mask.sum(dim=1)[:, None]
    assert objects_are_equal(masked_mean_along_seq(tensor, mask), expected)


###########################################
#     Tests for masked_amax_along_seq     #
###########################################


def test_masked_amax_along_seq() -> None:
    mask = torch.tensor([[True, True, False], [False, True, True]])
    assert objects_are_equal(masked_amax_along_seq(make_tensor(), mask), torch.tensor([2.0, 6.0]))


def test_masked_amax_along_seq_ignores_padded_larger_values() -> None:
    assert objects_are_equal(
        masked_amax_along_seq(make_tensor(), make_mask()), torch.tensor([2.0, 4.0])
    )


def test_masked_amax_along_seq_negative_values() -> None:
    assert objects_are_equal(
        masked_amax_along_seq(-make_tensor(), make_mask()), torch.tensor([-1.0, -4.0])
    )


def test_masked_amax_along_seq_all_false_row_float() -> None:
    mask = torch.tensor([[False, False, False], [True, True, True]])
    assert objects_are_equal(
        masked_amax_along_seq(make_tensor(), mask),
        torch.tensor([torch.finfo(torch.float32).min, 6.0]),
    )


def test_masked_amax_along_seq_all_false_row_int() -> None:
    mask = torch.tensor([[False, False, False], [True, True, True]])
    assert objects_are_equal(
        masked_amax_along_seq(torch.tensor([[1, 2, 3], [4, 5, 6]]), mask),
        torch.tensor([torch.iinfo(torch.int64).min, 6]),
    )


def test_masked_amax_along_seq_bool() -> None:
    tensor = torch.tensor([[True, False], [False, False]])
    mask = torch.tensor([[False, True], [True, True]])
    assert objects_are_equal(masked_amax_along_seq(tensor, mask), torch.tensor([False, False]))


def test_masked_amax_along_seq_keepdim() -> None:
    assert objects_are_equal(
        masked_amax_along_seq(make_tensor(), make_mask(), keepdim=True),
        torch.tensor([[2.0], [4.0]]),
    )


def test_masked_amax_along_seq_trailing_dims() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert objects_are_equal(
        masked_amax_along_seq(tensor, make_mask()), torch.tensor([[2.0, 3.0], [6.0, 7.0]])
    )


###########################################
#     Tests for masked_amin_along_seq     #
###########################################


def test_masked_amin_along_seq() -> None:
    mask = torch.tensor([[False, True, True], [True, True, False]])
    assert objects_are_equal(masked_amin_along_seq(make_tensor(), mask), torch.tensor([2.0, 4.0]))


def test_masked_amin_along_seq_ignores_padded_smaller_values() -> None:
    mask = torch.tensor([[False, True, True], [False, False, True]])
    assert objects_are_equal(masked_amin_along_seq(make_tensor(), mask), torch.tensor([2.0, 6.0]))


def test_masked_amin_along_seq_all_false_row_float() -> None:
    mask = torch.tensor([[False, False, False], [True, True, True]])
    assert objects_are_equal(
        masked_amin_along_seq(make_tensor(), mask),
        torch.tensor([torch.finfo(torch.float32).max, 4.0]),
    )


def test_masked_amin_along_seq_all_false_row_int() -> None:
    mask = torch.tensor([[False, False, False], [True, True, True]])
    assert objects_are_equal(
        masked_amin_along_seq(torch.tensor([[1, 2, 3], [4, 5, 6]]), mask),
        torch.tensor([torch.iinfo(torch.int64).max, 4]),
    )


def test_masked_amin_along_seq_bool() -> None:
    tensor = torch.tensor([[True, False], [True, True]])
    mask = torch.tensor([[True, False], [True, True]])
    assert objects_are_equal(masked_amin_along_seq(tensor, mask), torch.tensor([True, True]))


def test_masked_amin_along_seq_keepdim() -> None:
    assert objects_are_equal(
        masked_amin_along_seq(make_tensor(), make_mask(), keepdim=True),
        torch.tensor([[1.0], [4.0]]),
    )


def test_masked_amin_along_seq_trailing_dims() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert objects_are_equal(
        masked_amin_along_seq(tensor, make_mask()), torch.tensor([[0.0, 1.0], [6.0, 7.0]])
    )


##########################################################
#     Tests shared by the masked reductions (checks)     #
##########################################################


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_non_bool_mask_raises(fn: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"mask must be a boolean tensor but received dtype"):
        fn(make_tensor(), torch.ones(2, 3))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_1d_mask_raises(fn: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"mask must be a 2-D tensor but received a tensor with 1"):
        fn(make_tensor(), torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_3d_mask_raises(fn: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"mask must be a 2-D tensor but received a tensor with 3"):
        fn(make_tensor(), torch.ones(2, 3, 1, dtype=torch.bool))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_wrong_shape_mask_raises(fn: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(
        ValueError,
        match=r"mask must have the shape \(batch_size, seq_len\) of the tensor \(2, 3\) but received \(2, 4\)",
    ):
        fn(make_tensor(), torch.ones(2, 4, dtype=torch.bool))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_1d_tensor_raises(fn: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"mask must have the shape"):
        fn(torch.ones(2), torch.ones(2, 1, dtype=torch.bool))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_does_not_modify_tensor(fn: Callable[..., torch.Tensor]) -> None:
    tensor = make_tensor()
    fn(tensor, make_mask())
    assert objects_are_equal(tensor, make_tensor())


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_does_not_modify_mask(fn: Callable[..., torch.Tensor]) -> None:
    mask = make_mask()
    fn(make_tensor(), mask)
    assert objects_are_equal(mask, make_mask())


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_is_deterministic(fn: Callable[..., torch.Tensor]) -> None:
    assert objects_are_equal(fn(make_tensor(), make_mask()), fn(make_tensor(), make_mask()))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_mask_from_lengths_to_mask(fn: Callable[..., torch.Tensor]) -> None:
    mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
    assert objects_are_equal(fn(make_tensor(), mask), fn(make_tensor(), make_mask()))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS)
def test_masked_autograd(fn: Callable[..., torch.Tensor]) -> None:
    tensor = make_tensor().requires_grad_()
    fn(tensor, make_mask()).sum().backward()
    assert tensor.grad is not None


def test_masked_sum_along_seq_gradient() -> None:
    tensor = make_tensor().requires_grad_()
    masked_sum_along_seq(tensor, make_mask()).sum().backward()
    assert objects_are_equal(tensor.grad, torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def test_masked_mean_along_seq_gradient() -> None:
    tensor = make_tensor().requires_grad_()
    masked_mean_along_seq(tensor, make_mask()).sum().backward()
    assert objects_are_equal(tensor.grad, torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]))


#########################################
#     Tests for the private helpers     #
#########################################


def test_fill_invalid() -> None:
    assert objects_are_equal(
        _fill_invalid(make_tensor(), make_mask(), -1.0),
        torch.tensor([[1.0, 2.0, -1.0], [4.0, -1.0, -1.0]]),
    )


def test_fill_invalid_trailing_dims() -> None:
    out = _fill_invalid(torch.ones(2, 3, 2), make_mask(), 0.0)
    assert objects_are_equal(
        out,
        torch.tensor([[[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]),
    )


def test_fill_invalid_returns_a_new_tensor() -> None:
    tensor = make_tensor()
    _fill_invalid(tensor, make_mask(), 0.0)
    assert objects_are_equal(tensor, make_tensor())


def test_fill_invalid_non_bool_mask() -> None:
    with pytest.raises(ValueError, match=r"mask must be a boolean tensor"):
        _fill_invalid(make_tensor(), torch.ones(2, 3), 0.0)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.bool, False),
        (torch.float32, -3.4028234663852886e38),
        (torch.float64, -1.7976931348623157e308),
        (torch.float16, -65504.0),
        (torch.int8, -128),
        (torch.int64, -9223372036854775808),
        (torch.uint8, 0),
    ],
)
def test_lowest(dtype: torch.dtype, expected: float) -> None:
    assert _lowest(dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.bool, True),
        (torch.float32, 3.4028234663852886e38),
        (torch.float64, 1.7976931348623157e308),
        (torch.float16, 65504.0),
        (torch.int8, 127),
        (torch.int64, 9223372036854775807),
        (torch.uint8, 255),
    ],
)
def test_highest(dtype: torch.dtype, expected: float) -> None:
    assert _highest(dtype) == expected
