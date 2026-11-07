from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch
from coola.equality import objects_are_equal
from coola.random import make_torch_generator

from batchtensor.nested import (
    flip_along_batch,
    flip_along_seq,
    permute_along_batch,
    permute_along_seq,
    roll_along_batch,
    roll_along_seq,
    shuffle_along_batch,
    shuffle_along_seq,
)

if TYPE_CHECKING:
    from batchtensor.nested import NestedTensor

INDEX_DTYPES = [torch.int, torch.long]

#########################################
#     Tests for permute_along_batch     #
#########################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_batch_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        permute_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            torch.tensor([4, 3, 2, 1, 0], dtype=dtype),
        ),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_batch_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        permute_along_batch(
            {
                "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": torch.tensor([4, 3, 2, 1, 0]),
            },
            torch.tensor([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
            "b": torch.tensor([0, 1, 2, 3, 4]),
        },
    )


def test_permute_along_batch_nested() -> None:
    assert objects_are_equal(
        permute_along_batch(
            {
                "a": torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]),
                "b": torch.tensor([4, 3, 2, 1, 0], dtype=torch.float),
                "list": [torch.tensor([5, 6, 7, 8, 9])],
            },
            permutation=torch.tensor([2, 4, 1, 3, 0]),
        ),
        {
            "a": torch.tensor([[2, 5], [3, 8], [1, 7], [5, 6], [4, 9]]),
            "b": torch.tensor([2, 0, 3, 1, 4], dtype=torch.float),
            "list": [torch.tensor([7, 9, 6, 8, 5])],
        },
    )


def test_permute_along_batch_incorrect_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"permutation shape \(.*\) is not compatible with tensor shape \(.*\)",
    ):
        permute_along_batch(
            {
                "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": torch.tensor([4, 3, 2, 1, 0]),
            },
            torch.tensor([4, 3, 2, 1, 0, 2]),
        )


#######################################
#     Tests for permute_along_seq     #
#######################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_seq_tensor(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        permute_along_seq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            torch.tensor([4, 3, 2, 1, 0], dtype=dtype),
        ),
        torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_seq_dict(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        permute_along_seq(
            {
                "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": torch.tensor([[4, 3, 2, 1, 0]]),
            },
            torch.tensor([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": torch.tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
            "b": torch.tensor([[0, 1, 2, 3, 4]]),
        },
    )


def test_permute_along_seq_nested() -> None:
    assert objects_are_equal(
        permute_along_seq(
            {
                "a": torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]),
                "b": torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.float),
                "list": [torch.tensor([[5, 6, 7, 8, 9]])],
            },
            permutation=torch.tensor([2, 4, 1, 3, 0]),
        ),
        {
            "a": torch.tensor([[2, 3, 1, 5, 4], [5, 8, 7, 6, 9]]),
            "b": torch.tensor([[2, 0, 3, 1, 4]], dtype=torch.float),
            "list": [torch.tensor([[7, 9, 6, 8, 5]])],
        },
    )


def test_permute_along_seq_incorrect_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"permutation shape \(.*\) is not compatible with tensor shape \(.*\)",
    ):
        permute_along_seq(
            {
                "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": torch.tensor([[4, 3, 2, 1, 0]]),
            },
            torch.tensor([4, 3, 2, 1, 0, 2]),
        )


#########################################
#     Tests for shuffle_along_batch     #
#########################################


@patch(
    "batchtensor.utils.seed.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]),  # noqa: ARG005
)
def test_shuffle_along_batch_tensor() -> None:
    assert objects_are_equal(
        shuffle_along_batch(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])),
        torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]),
    )


@patch(
    "batchtensor.utils.seed.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_shuffle_along_batch_dict() -> None:
    assert objects_are_equal(
        shuffle_along_batch(
            {
                "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": torch.tensor([4, 3, 2, 1, 0]),
            }
        ),
        {
            "a": torch.tensor([[4, 5], [8, 9], [2, 3], [6, 7], [0, 1]]),
            "b": torch.tensor([2, 0, 3, 1, 4]),
        },
    )


@patch(
    "batchtensor.utils.seed.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_shuffle_along_batch_nested() -> None:
    assert objects_are_equal(
        shuffle_along_batch(
            {
                "a": torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]),
                "b": torch.tensor([4, 3, 2, 1, 0], dtype=torch.float),
                "list": [torch.tensor([5, 6, 7, 8, 9])],
            }
        ),
        {
            "a": torch.tensor([[2, 5], [3, 8], [1, 7], [5, 6], [4, 9]]),
            "b": torch.tensor([2, 0, 3, 1, 4], dtype=torch.float),
            "list": [torch.tensor([7, 9, 6, 8, 5])],
        },
    )


def test_shuffle_along_batch_same_random_seed() -> None:
    data = {
        "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        "b": torch.tensor([4, 3, 2, 1, 0]),
    }
    assert objects_are_equal(
        shuffle_along_batch(data, generator=make_torch_generator(1)),
        shuffle_along_batch(data, generator=make_torch_generator(1)),
    )


def test_shuffle_along_batch_different_random_seeds() -> None:
    data = {
        "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        "b": torch.tensor([4, 3, 2, 1, 0]),
    }
    assert not objects_are_equal(
        shuffle_along_batch(data, generator=make_torch_generator(1)),
        shuffle_along_batch(data, generator=make_torch_generator(2)),
    )


def test_shuffle_along_batch_multiple_shuffle() -> None:
    data = {
        "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        "b": torch.tensor([4, 3, 2, 1, 0]),
    }
    generator = make_torch_generator(1)
    assert not objects_are_equal(
        shuffle_along_batch(data, generator=generator),
        shuffle_along_batch(data, generator=generator),
    )


#######################################
#     Tests for shuffle_along_seq     #
#######################################


@patch(
    "batchtensor.utils.seed.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_shuffle_along_seq_tensor() -> None:
    assert objects_are_equal(
        shuffle_along_seq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])),
        torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]]),
    )


@patch(
    "batchtensor.utils.seed.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_shuffle_along_seq_dict() -> None:
    assert objects_are_equal(
        shuffle_along_seq(
            {
                "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": torch.tensor([[4, 3, 2, 1, 0]]),
            }
        ),
        {
            "a": torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]]),
            "b": torch.tensor([[2, 0, 3, 1, 4]]),
        },
    )


@patch(
    "batchtensor.utils.seed.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_shuffle_along_seq_nested() -> None:
    assert objects_are_equal(
        shuffle_along_seq(
            {
                "a": torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]),
                "b": torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.float),
                "list": [torch.tensor([[5, 6, 7, 8, 9]])],
            }
        ),
        {
            "a": torch.tensor([[2, 3, 1, 5, 4], [5, 8, 7, 6, 9]]),
            "b": torch.tensor([[2, 0, 3, 1, 4]], dtype=torch.float),
            "list": [torch.tensor([[7, 9, 6, 8, 5]])],
        },
    )


def test_shuffle_along_seq_same_random_seed() -> None:
    data = {
        "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        "b": torch.tensor([[4, 3, 2, 1, 0]]),
    }
    assert objects_are_equal(
        shuffle_along_seq(data, generator=make_torch_generator(1)),
        shuffle_along_seq(data, generator=make_torch_generator(1)),
    )


def test_shuffle_along_seq_different_random_seeds() -> None:
    data = {
        "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        "b": torch.tensor([[4, 3, 2, 1, 0]]),
    }
    assert not objects_are_equal(
        shuffle_along_seq(data, generator=make_torch_generator(1)),
        shuffle_along_seq(data, generator=make_torch_generator(2)),
    )


def test_shuffle_along_seq_multiple_shuffle() -> None:
    data = {
        "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        "b": torch.tensor([[4, 3, 2, 1, 0]]),
    }
    generator = make_torch_generator(1)
    assert not objects_are_equal(
        shuffle_along_seq(data, generator=generator), shuffle_along_seq(data, generator=generator)
    )


def test_shuffle_along_batch_empty() -> None:
    with pytest.raises(ValueError, match="does not contain any tensor"):
        shuffle_along_batch({})


def test_shuffle_along_seq_empty() -> None:
    with pytest.raises(ValueError, match="does not contain any tensor"):
        shuffle_along_seq([])


def test_shuffle_along_batch_size_mismatch() -> None:
    with pytest.raises(ValueError, match="same size"):
        shuffle_along_batch({"a": torch.ones(3, 2), "b": torch.ones(4)})


def test_shuffle_along_batch_int_seed_same_as_generator() -> None:
    data: NestedTensor = {"a": torch.arange(12).reshape(4, 3), "b": [torch.arange(4)]}
    assert objects_are_equal(
        shuffle_along_batch(data, generator=7),
        shuffle_along_batch(data, generator=make_torch_generator(7)),
    )


def test_shuffle_along_batch_int_seed_repeat_call() -> None:
    data: NestedTensor = {"a": torch.arange(12).reshape(4, 3), "b": [torch.arange(4)]}
    assert objects_are_equal(
        shuffle_along_batch(data, generator=7), shuffle_along_batch(data, generator=7)
    )


def test_shuffle_along_seq_int_seed_same_as_generator() -> None:
    data: NestedTensor = {
        "a": torch.arange(12).reshape(2, 6),
        "b": [torch.arange(12).reshape(2, 6)],
    }
    assert objects_are_equal(
        shuffle_along_seq(data, generator=7),
        shuffle_along_seq(data, generator=make_torch_generator(7)),
    )


def test_shuffle_along_seq_int_seed_repeat_call() -> None:
    data: NestedTensor = {
        "a": torch.arange(12).reshape(2, 6),
        "b": [torch.arange(12).reshape(2, 6)],
    }
    assert objects_are_equal(
        shuffle_along_seq(data, generator=7), shuffle_along_seq(data, generator=7)
    )


#########################################################
#     Tests for flip_along_batch and flip_along_seq     #
#########################################################


def test_flip_along_batch_tensor() -> None:
    assert objects_are_equal(
        flip_along_batch(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])),
        torch.tensor([[8, 7, 2, 1], [4, 9, 5, 6], [0, 1, 3, 2]]),
    )


def test_flip_along_batch_dict_and_list() -> None:
    assert objects_are_equal(
        flip_along_batch(
            {
                "a": torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),
                "b": [torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])],
            }
        ),
        {
            "a": torch.tensor([[8, 7, 2, 1], [4, 9, 5, 6], [0, 1, 3, 2]]),
            "b": [torch.tensor([[8, 7, 2, 1], [4, 9, 5, 6], [0, 1, 3, 2]])],
        },
    )


def test_flip_along_batch_tuple() -> None:
    out = flip_along_batch((torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),))
    assert isinstance(out, tuple)


def test_flip_along_batch_non_tensor_leaf() -> None:
    data: Any = {"a": "x"}
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        flip_along_batch(data)


def test_flip_along_seq_tensor() -> None:
    assert objects_are_equal(
        flip_along_seq(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])),
        torch.tensor([[2, 3, 1, 0], [6, 5, 9, 4], [1, 2, 7, 8]]),
    )


def test_flip_along_seq_dict_and_list() -> None:
    assert objects_are_equal(
        flip_along_seq(
            {
                "a": torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),
                "b": [torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])],
            }
        ),
        {
            "a": torch.tensor([[2, 3, 1, 0], [6, 5, 9, 4], [1, 2, 7, 8]]),
            "b": [torch.tensor([[2, 3, 1, 0], [6, 5, 9, 4], [1, 2, 7, 8]])],
        },
    )


def test_flip_along_seq_tuple() -> None:
    out = flip_along_seq((torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),))
    assert isinstance(out, tuple)


def test_flip_along_seq_non_tensor_leaf() -> None:
    data: Any = {"a": "x"}
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        flip_along_seq(data)


#########################################################
#     Tests for roll_along_batch and roll_along_seq     #
#########################################################


def test_roll_along_batch_tensor() -> None:
    assert objects_are_equal(
        roll_along_batch(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 1),
        torch.tensor([[8, 7, 2, 1], [0, 1, 3, 2], [4, 9, 5, 6]]),
    )


def test_roll_along_batch_dict_and_list() -> None:
    assert objects_are_equal(
        roll_along_batch(
            {
                "a": torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),
                "b": [torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])],
            },
            1,
        ),
        {
            "a": torch.tensor([[8, 7, 2, 1], [0, 1, 3, 2], [4, 9, 5, 6]]),
            "b": [torch.tensor([[8, 7, 2, 1], [0, 1, 3, 2], [4, 9, 5, 6]])],
        },
    )


def test_roll_along_batch_tuple() -> None:
    out = roll_along_batch((torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),), 1)
    assert isinstance(out, tuple)


def test_roll_along_batch_non_tensor_leaf() -> None:
    data: Any = {"a": "x"}
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        roll_along_batch(data, 1)


def test_roll_along_seq_tensor() -> None:
    assert objects_are_equal(
        roll_along_seq(torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]), 1),
        torch.tensor([[2, 0, 1, 3], [6, 4, 9, 5], [1, 8, 7, 2]]),
    )


def test_roll_along_seq_dict_and_list() -> None:
    assert objects_are_equal(
        roll_along_seq(
            {
                "a": torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),
                "b": [torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])],
            },
            1,
        ),
        {
            "a": torch.tensor([[2, 0, 1, 3], [6, 4, 9, 5], [1, 8, 7, 2]]),
            "b": [torch.tensor([[2, 0, 1, 3], [6, 4, 9, 5], [1, 8, 7, 2]])],
        },
    )


def test_roll_along_seq_tuple() -> None:
    out = roll_along_seq((torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]]),), 1)
    assert isinstance(out, tuple)


def test_roll_along_seq_non_tensor_leaf() -> None:
    data: Any = {"a": "x"}
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        roll_along_seq(data, 1)
