from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.nested import (
    amax_along_batch,
    amax_along_seq,
    amin_along_batch,
    amin_along_seq,
)
from tests.unit.tensor.test_reduction import DTYPES

######################################
#     Tests for amax_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_tensor_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[8, 9]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(
            {"a": torch.arange(10, dtype=dtype).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
        ),
        {"a": torch.tensor([8, 9], dtype=dtype), "b": torch.tensor(4)},
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_dict_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(
            {"a": torch.arange(10, dtype=dtype).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])},
            keepdim=True,
        ),
        {"a": torch.tensor([[8, 9]], dtype=dtype), "b": torch.tensor([4])},
    )


def test_amax_along_batch_nested() -> None:
    assert objects_are_equal(
        amax_along_batch(
            {
                "a": torch.arange(10).view(5, 2),
                "b": torch.tensor([4, 3, 2, 1, 0]),
                "c": [torch.tensor([5, 6, 7, 8, 9])],
            }
        ),
        {"a": torch.tensor([8, 9]), "b": torch.tensor(4), "c": [torch.tensor(9)]},
    )


####################################
#     Tests for amax_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_tensor_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[4], [9]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(
            {"a": torch.arange(10, dtype=dtype).view(2, 5), "b": torch.tensor([[4, 3, 2, 1, 0]])}
        ),
        {"a": torch.tensor([4, 9], dtype=dtype), "b": torch.tensor([4])},
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_dict_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(
            {"a": torch.arange(10, dtype=dtype).view(2, 5), "b": torch.tensor([[4, 3, 2, 1, 0]])},
            keepdim=True,
        ),
        {"a": torch.tensor([[4], [9]], dtype=dtype), "b": torch.tensor([[4]])},
    )


def test_amax_along_seq_nested() -> None:
    assert objects_are_equal(
        amax_along_seq(
            {
                "a": torch.arange(10).view(2, 5),
                "b": torch.tensor([[4, 3, 2, 1, 0]]),
                "c": [torch.tensor([[5, 6, 7, 8, 9]])],
            }
        ),
        {"a": torch.tensor([4, 9]), "b": torch.tensor([4]), "c": [torch.tensor([9])]},
    )


######################################
#     Tests for amin_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([0, 1], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_tensor_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[0, 1]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(
            {"a": torch.arange(10, dtype=dtype).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
        ),
        {"a": torch.tensor([0, 1], dtype=dtype), "b": torch.tensor(0)},
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_dict_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(
            {"a": torch.arange(10, dtype=dtype).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])},
            keepdim=True,
        ),
        {"a": torch.tensor([[0, 1]], dtype=dtype), "b": torch.tensor([0])},
    )


def test_amin_along_batch_nested() -> None:
    assert objects_are_equal(
        amin_along_batch(
            {
                "a": torch.arange(10).view(5, 2),
                "b": torch.tensor([4, 3, 2, 1, 0]),
                "c": [torch.tensor([5, 6, 7, 8, 9])],
            }
        ),
        {"a": torch.tensor([0, 1]), "b": torch.tensor(0), "c": [torch.tensor(5)]},
    )


####################################
#     Tests for amin_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([0, 5], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_tensor_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[0], [5]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(
            {"a": torch.arange(10, dtype=dtype).view(2, 5), "b": torch.tensor([[4, 3, 2, 1, 0]])}
        ),
        {"a": torch.tensor([0, 5], dtype=dtype), "b": torch.tensor([0])},
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_dict_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(
            {"a": torch.arange(10, dtype=dtype).view(2, 5), "b": torch.tensor([[4, 3, 2, 1, 0]])},
            keepdim=True,
        ),
        {"a": torch.tensor([[0], [5]], dtype=dtype), "b": torch.tensor([[0]])},
    )


def test_amin_along_seq_nested() -> None:
    assert objects_are_equal(
        amin_along_seq(
            {
                "a": torch.arange(10).view(2, 5),
                "b": torch.tensor([[4, 3, 2, 1, 0]]),
                "c": [torch.tensor([[5, 6, 7, 8, 9]])],
            }
        ),
        {"a": torch.tensor([0, 5]), "b": torch.tensor([0]), "c": [torch.tensor([5])]},
    )
