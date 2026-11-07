from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor import nested, tensor

if TYPE_CHECKING:
    from batchtensor.nested import NestedTensor

REDUCTIONS = ["sum", "mean", "prod", "amax", "amin", "argmax", "argmin", "max", "min", "median"]

# (function name, keyword arguments) for the functions that map a tensor to a
# tensor (or a tuple of tensors for ``chunk`` and ``split``).
CASES = (
    [(f"{name}_along_{dim}", {}) for name in REDUCTIONS for dim in ("batch", "seq")]
    + [
        (f"{name}_along_{dim}", {"keepdim": True})
        for name in REDUCTIONS
        for dim in ("batch", "seq")
    ]
    + [
        (f"{name}_along_{dim}", kwargs)
        for name in ("sort", "argsort")
        for dim in ("batch", "seq")
        for kwargs in ({}, {"descending": True, "stable": True})
    ]
    + [(f"{name}_along_{dim}", {}) for name in ("cumsum", "cumprod") for dim in ("batch", "seq")]
    + [(f"slice_along_{dim}", {"start": 1, "stop": 3, "step": 1}) for dim in ("batch", "seq")]
    + [(f"select_along_{dim}", {"index": 1}) for dim in ("batch", "seq")]
    + [(f"chunk_along_{dim}", {"chunks": 2}) for dim in ("batch", "seq")]
    + [(f"split_along_{dim}", {"split_size_or_sections": 2}) for dim in ("batch", "seq")]
    + [(f"repeat_along_{dim}", {"repeats": 2}) for dim in ("batch", "seq")]
    + [(f"index_select_along_{dim}", {"index": torch.tensor([0, 2])}) for dim in ("batch", "seq")]
    + [
        ("permute_along_batch", {"permutation": torch.tensor([3, 1, 0, 2])}),
        ("permute_along_seq", {"permutation": torch.tensor([4, 3, 2, 1, 0])}),
    ]
    + [(f"shuffle_along_{dim}", {"generator": 0}) for dim in ("batch", "seq")]
    + [
        (f"std_along_{dim}", variant)
        for dim in ("batch", "seq")
        for variant in [{}, {"keepdim": True}, {"correction": 0}]
    ]
    + [
        (f"var_along_{dim}", variant)
        for dim in ("batch", "seq")
        for variant in [{}, {"keepdim": True}, {"correction": 0}]
    ]
    + [
        (f"any_along_{dim}", variant)
        for dim in ("batch", "seq")
        for variant in [{}, {"keepdim": True}]
    ]
    + [
        (f"all_along_{dim}", variant)
        for dim in ("batch", "seq")
        for variant in [{}, {"keepdim": True}]
    ]
    + [
        (f"pad_along_{dim}", {"pad_size": 2, "value": 1.0, "side": side})
        for dim in ("batch", "seq")
        for side in ("left", "right")
    ]
)
IDS = [f"{name}-{i}" for i, (name, _) in enumerate(CASES)]


def make_tensor() -> torch.Tensor:
    return torch.arange(60, dtype=torch.float32).reshape(4, 5, 3).sin()


def make_nested() -> NestedTensor:
    x = make_tensor()
    return {"a": x, "b": [x, x]}


def to_nested_result(name: str, result: Any) -> Any:
    r"""Build the expected nested result from the tensor result."""
    if name.startswith(("chunk_", "split_")):
        return tuple({"a": item, "b": [item, item]} for item in result)
    return {"a": result, "b": [result, result]}


@pytest.mark.parametrize(("name", "kwargs"), CASES, ids=IDS)
def test_nested_tensor_equals_tensor(name: str, kwargs: dict[str, Any]) -> None:
    expected = getattr(tensor, name)(make_tensor(), **kwargs)
    assert objects_are_equal(getattr(nested, name)(make_tensor(), **kwargs), expected)


@pytest.mark.parametrize(("name", "kwargs"), CASES, ids=IDS)
def test_nested_structure_equals_tensor(name: str, kwargs: dict[str, Any]) -> None:
    expected = to_nested_result(name, getattr(tensor, name)(make_tensor(), **kwargs))
    assert objects_are_equal(getattr(nested, name)(make_nested(), **kwargs), expected)


@pytest.mark.parametrize(
    "name", ["cat_along_batch", "cat_along_seq", "stack_along_batch", "stack_along_seq"]
)
def test_joining_nested_equals_tensor(name: str) -> None:
    tensors = [make_tensor(), make_tensor() + 1]
    expected = getattr(tensor, name)(tensors)
    out = getattr(nested, name)([{"a": t, "b": [t]} for t in tensors])
    assert objects_are_equal(out, {"a": expected, "b": [expected]})
