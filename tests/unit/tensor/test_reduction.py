from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from batchtensor.tensor import (
    amax_along_batch,
    amax_along_seq,
    amin_along_batch,
    amin_along_seq,
    argmax_along_batch,
    argmax_along_seq,
    argmin_along_batch,
    argmin_along_seq,
    sum_along_batch,
    sum_along_seq,
)

DTYPES = (torch.float, torch.long)


######################################
#     Tests for amax_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[8, 9]], dtype=dtype),
    )


####################################
#     Tests for amax_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[4], [9]], dtype=dtype),
    )


######################################
#     Tests for amin_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([0, 1], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[0, 1]], dtype=dtype),
    )


####################################
#     Tests for amin_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([0, 5], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[0], [5]], dtype=dtype),
    )


########################################
#     Tests for argmax_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[4, 4]]),
    )


######################################
#     Tests for argmax_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[4], [4]]),
    )


########################################
#     Tests for argmin_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[0, 0]]),
    )


######################################
#     Tests for argmin_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[0], [0]]),
    )


#####################################
#     Tests for sum_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_batch(torch.arange(10, dtype=dtype).view(5, 2)),
        torch.tensor([20, 25], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_batch(torch.arange(10, dtype=dtype).view(5, 2), keepdim=True),
        torch.tensor([[20, 25]], dtype=dtype),
    )


###################################
#     Tests for sum_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_seq(torch.arange(10, dtype=dtype).view(2, 5)),
        torch.tensor([10, 35], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_seq(torch.arange(10, dtype=dtype).view(2, 5), keepdim=True),
        torch.tensor([[10], [35]], dtype=dtype),
    )
