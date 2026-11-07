from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.tensor import (
    amax_along_batch,
    amax_along_seq,
    amin_along_batch,
    amin_along_seq,
    any_along_batch,
    any_along_seq,
    argmax_along_batch,
    argmax_along_seq,
    argmin_along_batch,
    argmin_along_seq,
    max_along_batch,
    max_along_seq,
    mean_along_batch,
    mean_along_seq,
    median_along_batch,
    median_along_seq,
    min_along_batch,
    min_along_seq,
    prod_along_batch,
    prod_along_seq,
    std_along_batch,
    std_along_seq,
    sum_along_batch,
    sum_along_seq,
    var_along_batch,
    var_along_seq,
)

DTYPES = [torch.float, torch.double, torch.long]
FLOATING_DTYPES = [torch.float, torch.double]


######################################
#     Tests for amax_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[8, 9]], dtype=dtype),
    )


####################################
#     Tests for amax_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amax_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.tensor([[4], [9]], dtype=dtype),
    )


######################################
#     Tests for amin_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([0, 1], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[0, 1]], dtype=dtype),
    )


####################################
#     Tests for amin_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([0, 5], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        amin_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.tensor([[0], [5]], dtype=dtype),
    )


########################################
#     Tests for argmax_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[4, 4]]),
    )


######################################
#     Tests for argmax_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmax_along_seq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[4], [4]]),
    )


########################################
#     Tests for argmin_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[0, 0]]),
    )


######################################
#     Tests for argmin_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        argmin_along_seq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[0], [0]]),
    )


#####################################
#     Tests for max_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        max_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.return_types.max((torch.tensor([8, 9], dtype=dtype), torch.tensor([4, 4]))),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        max_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.return_types.max((torch.tensor([[8, 9]], dtype=dtype), torch.tensor([[4, 4]]))),
    )


###################################
#     Tests for max_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        max_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.return_types.max((torch.tensor([4, 9], dtype=dtype), torch.tensor([4, 4]))),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        max_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.return_types.max((torch.tensor([[4], [9]], dtype=dtype), torch.tensor([[4], [4]]))),
    )


######################################
#     Tests for mean_along_batch     #
######################################


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        mean_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([4.0, 5.0], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        mean_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[4.0, 5.0]], dtype=dtype),
    )


####################################
#     Tests for mean_along_seq     #
####################################


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        mean_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([2.0, 7.0], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        mean_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.tensor([[2.0], [7.0]], dtype=dtype),
    )


########################################
#     Tests for median_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        median_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.return_types.median((torch.tensor([4, 5], dtype=dtype), torch.tensor([2, 2]))),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        median_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.return_types.median((torch.tensor([[4, 5]], dtype=dtype), torch.tensor([[2, 2]]))),
    )


######################################
#     Tests for median_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        median_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.return_types.median((torch.tensor([2, 7], dtype=dtype), torch.tensor([2, 2]))),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        median_along_seq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True
        ),
        torch.return_types.median(
            (torch.tensor([[2], [7]], dtype=dtype), torch.tensor([[2], [2]]))
        ),
    )


#####################################
#     Tests for min_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        min_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.return_types.min((torch.tensor([0, 1], dtype=dtype), torch.tensor([0, 0]))),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        min_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.return_types.min((torch.tensor([[0, 1]], dtype=dtype), torch.tensor([[0, 0]]))),
    )


###################################
#     Tests for min_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        min_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.return_types.min((torch.tensor([0, 5], dtype=dtype), torch.tensor([0, 0]))),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        min_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.return_types.min((torch.tensor([[0], [5]], dtype=dtype), torch.tensor([[0], [0]]))),
    )


######################################
#     Tests for prod_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        prod_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([0, 945], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        prod_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[0, 945]], dtype=dtype),
    )


####################################
#     Tests for prod_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        prod_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([0, 15120], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        prod_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.tensor([[0], [15120]], dtype=dtype),
    )


#####################################
#     Tests for sum_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        torch.tensor([20, 25], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdim=True
        ),
        torch.tensor([[20, 25]], dtype=dtype),
    )


###################################
#     Tests for sum_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        torch.tensor([10, 35], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        sum_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdim=True),
        torch.tensor([[10], [35]], dtype=dtype),
    )


#######################################################
#     Tests for std_along_batch and std_along_seq     #
#######################################################


def test_std_along_batch_default() -> None:
    assert objects_are_equal(
        std_along_batch(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
        ),
        torch.tensor([4.0, 4.163331985473633, 1.5275251865386963, 2.6457512378692627]),
    )


def test_std_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        std_along_batch(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            keepdim=True,
        ),
        torch.tensor([[4.0, 4.163331985473633, 1.5275251865386963, 2.6457512378692627]]),
    )


def test_std_along_batch_correction_0() -> None:
    assert objects_are_equal(
        std_along_batch(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            correction=0,
        ),
        torch.tensor(
            [3.265986442565918, 3.399346351623535, 1.2472190856933594, 2.1602468490600586]
        ),
    )


def test_std_along_batch_does_not_modify_input() -> None:
    tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    std_along_batch(tensor)
    assert objects_are_equal(
        tensor, torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    )


def test_std_along_seq_default() -> None:
    assert objects_are_equal(
        std_along_seq(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
        ),
        torch.tensor([1.29099440574646, 2.1602468490600586, 3.5118846893310547]),
    )


def test_std_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        std_along_seq(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            keepdim=True,
        ),
        torch.tensor([[1.29099440574646], [2.1602468490600586], [3.5118846893310547]]),
    )


def test_std_along_seq_correction_0() -> None:
    assert objects_are_equal(
        std_along_seq(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            correction=0,
        ),
        torch.tensor([1.1180340051651, 1.8708287477493286, 3.041381359100342]),
    )


def test_std_along_seq_does_not_modify_input() -> None:
    tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    std_along_seq(tensor)
    assert objects_are_equal(
        tensor, torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    )


def test_std_along_seq_1d_tensor() -> None:
    with pytest.raises(IndexError, match="Dimension out of range"):
        std_along_seq(torch.tensor([0.0, 4.0, 8.0]))


#######################################################
#     Tests for var_along_batch and var_along_seq     #
#######################################################


def test_var_along_batch_default() -> None:
    assert objects_are_equal(
        var_along_batch(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
        ),
        torch.tensor([16.0, 17.33333396911621, 2.3333332538604736, 7.0]),
    )


def test_var_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        var_along_batch(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            keepdim=True,
        ),
        torch.tensor([[16.0, 17.33333396911621, 2.3333332538604736, 7.0]]),
    )


def test_var_along_batch_correction_0() -> None:
    assert objects_are_equal(
        var_along_batch(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            correction=0,
        ),
        torch.tensor(
            [10.666666984558105, 11.55555534362793, 1.5555555820465088, 4.666666507720947]
        ),
    )


def test_var_along_batch_does_not_modify_input() -> None:
    tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    var_along_batch(tensor)
    assert objects_are_equal(
        tensor, torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    )


def test_var_along_seq_default() -> None:
    assert objects_are_equal(
        var_along_seq(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
        ),
        torch.tensor([1.6666666269302368, 4.666666507720947, 12.333333015441895]),
    )


def test_var_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        var_along_seq(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            keepdim=True,
        ),
        torch.tensor([[1.6666666269302368], [4.666666507720947], [12.333333015441895]]),
    )


def test_var_along_seq_correction_0() -> None:
    assert objects_are_equal(
        var_along_seq(
            torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]]),
            correction=0,
        ),
        torch.tensor([1.25, 3.5, 9.25]),
    )


def test_var_along_seq_does_not_modify_input() -> None:
    tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    var_along_seq(tensor)
    assert objects_are_equal(
        tensor, torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
    )


def test_var_along_seq_1d_tensor() -> None:
    with pytest.raises(IndexError, match="Dimension out of range"):
        var_along_seq(torch.tensor([0.0, 4.0, 8.0]))


#######################################################
#     Tests for any_along_batch and any_along_seq     #
#######################################################


def test_any_along_batch_default() -> None:
    assert objects_are_equal(
        any_along_batch(
            torch.tensor(
                [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
            )
        ),
        torch.tensor([True, True, True, True]),
    )


def test_any_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        any_along_batch(
            torch.tensor(
                [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
            ),
            keepdim=True,
        ),
        torch.tensor([[True, True, True, True]]),
    )


def test_any_along_batch_does_not_modify_input() -> None:
    tensor = torch.tensor(
        [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
    )
    any_along_batch(tensor)
    assert objects_are_equal(
        tensor,
        torch.tensor(
            [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
        ),
    )


def test_any_along_seq_default() -> None:
    assert objects_are_equal(
        any_along_seq(
            torch.tensor(
                [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
            )
        ),
        torch.tensor([True, True, True]),
    )


def test_any_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        any_along_seq(
            torch.tensor(
                [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
            ),
            keepdim=True,
        ),
        torch.tensor([[True], [True], [True]]),
    )


def test_any_along_seq_does_not_modify_input() -> None:
    tensor = torch.tensor(
        [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
    )
    any_along_seq(tensor)
    assert objects_are_equal(
        tensor,
        torch.tensor(
            [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
        ),
    )


def test_any_along_seq_1d_tensor() -> None:
    with pytest.raises(IndexError, match="Dimension out of range"):
        any_along_seq(torch.tensor([True, True, False]))
