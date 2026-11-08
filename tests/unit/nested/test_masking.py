from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.nested import (
    masked_amax_along_seq,
    masked_amin_along_seq,
    masked_mean_along_seq,
    masked_sum_along_seq,
)

if TYPE_CHECKING:
    from collections.abc import Callable

MASKED_FUNCTIONS = [
    masked_sum_along_seq,
    masked_mean_along_seq,
    masked_amax_along_seq,
    masked_amin_along_seq,
]
EXPECTED = {
    "masked_sum_along_seq": torch.tensor([3.0, 4.0]),
    "masked_mean_along_seq": torch.tensor([1.5, 4.0]),
    "masked_amax_along_seq": torch.tensor([2.0, 4.0]),
    "masked_amin_along_seq": torch.tensor([1.0, 4.0]),
}
CASES = [(fn, EXPECTED[fn.__name__]) for fn in MASKED_FUNCTIONS]
IDS = [fn.__name__ for fn in MASKED_FUNCTIONS]


def make_tensor() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def make_mask() -> torch.Tensor:
    return torch.tensor([[True, True, False], [True, False, False]])


#########################################
#     Tests for the masked reductions   #
#########################################


@pytest.mark.parametrize(("fn", "expected"), CASES, ids=IDS)
def test_masked_tensor(fn: Callable[..., Any], expected: torch.Tensor) -> None:
    assert objects_are_equal(fn(make_tensor(), make_mask()), expected)


@pytest.mark.parametrize(("fn", "expected"), CASES, ids=IDS)
def test_masked_dict(fn: Callable[..., Any], expected: torch.Tensor) -> None:
    out = fn({"a": make_tensor(), "b": make_tensor()}, make_mask())
    assert objects_are_equal(out, {"a": expected, "b": expected})


@pytest.mark.parametrize(("fn", "expected"), CASES, ids=IDS)
def test_masked_list(fn: Callable[..., Any], expected: torch.Tensor) -> None:
    assert objects_are_equal(fn([make_tensor(), make_tensor()], make_mask()), [expected, expected])


@pytest.mark.parametrize(("fn", "expected"), CASES, ids=IDS)
def test_masked_tuple(fn: Callable[..., Any], expected: torch.Tensor) -> None:
    assert objects_are_equal(fn((make_tensor(), make_tensor()), make_mask()), (expected, expected))


@pytest.mark.parametrize(("fn", "expected"), CASES, ids=IDS)
def test_masked_nested_mix(fn: Callable[..., Any], expected: torch.Tensor) -> None:
    out = fn({"a": [make_tensor(), {"b": make_tensor()}]}, make_mask())
    assert objects_are_equal(out, {"a": [expected, {"b": expected}]})


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_ordered_dict(fn: Callable[..., Any]) -> None:
    out = fn(OrderedDict(a=make_tensor()), make_mask())
    assert type(out) is OrderedDict


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_shared_mask_is_used_by_all_the_leaves(fn: Callable[..., Any]) -> None:
    out = fn({"a": make_tensor(), "b": make_tensor().unsqueeze(-1)}, make_mask())
    assert objects_are_equal(out["a"], out["b"].squeeze(-1))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_keepdim(fn: Callable[..., Any]) -> None:
    out = fn({"a": make_tensor()}, make_mask(), keepdim=True)
    assert out["a"].shape == (2, 1)


def test_masked_sum_whole_output() -> None:
    out = masked_sum_along_seq(
        {"a": make_tensor(), "b": [torch.ones(2, 3, 2)]}, make_mask(), keepdim=True
    )
    assert objects_are_equal(
        out,
        {
            "a": torch.tensor([[3.0], [4.0]]),
            "b": [torch.tensor([[[2.0, 2.0]], [[1.0, 1.0]]])],
        },
    )


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_leaf_with_wrong_shape_raises(fn: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match=r"mask must have the shape"):
        fn({"a": make_tensor(), "b": torch.ones(2, 4)}, make_mask())


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_1d_leaf_raises(fn: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match=r"mask must have the shape"):
        fn({"a": torch.ones(2)}, make_mask())


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_non_bool_mask_raises(fn: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match=r"mask must be a boolean tensor"):
        fn({"a": make_tensor()}, torch.ones(2, 3))


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_non_tensor_leaf_raises(fn: Callable[..., Any]) -> None:
    data = cast("Any", {"a": make_tensor(), "b": "abc"})
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        fn(data, make_mask())


def test_masked_mean_int_leaf_raises() -> None:
    with pytest.raises(ValueError, match=r"requires a floating point or complex tensor"):
        masked_mean_along_seq({"a": torch.ones(2, 3, dtype=torch.long)}, make_mask())


@pytest.mark.parametrize("fn", MASKED_FUNCTIONS, ids=IDS)
def test_masked_does_not_modify_input(fn: Callable[..., Any]) -> None:
    data = {"a": make_tensor()}
    fn(data, make_mask())
    assert objects_are_equal(data, {"a": make_tensor()})
