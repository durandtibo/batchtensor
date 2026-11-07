from __future__ import annotations

from typing import Any

import pytest
import torch

from batchtensor import nested

# Every nested function that maps or splits a batch raises the same TypeError
# for a leaf that is not a tensor, with the real collaborators (no patching).
REDUCTIONS = ["sum", "mean", "prod", "amax", "amin", "argmax", "argmin", "max", "min", "median"]
CASES: list[tuple[str, dict[str, Any]]] = (
    [(f"{name}_along_{dim}", {}) for name in REDUCTIONS for dim in ("batch", "seq")]
    + [
        (f"{name}_along_{dim}", {})
        for name in ("sort", "argsort", "cumsum", "cumprod")
        for dim in ("batch", "seq")
    ]
    + [(f"slice_along_{dim}", {}) for dim in ("batch", "seq")]
    + [(f"select_along_{dim}", {"index": 0}) for dim in ("batch", "seq")]
    + [(f"chunk_along_{dim}", {"chunks": 2}) for dim in ("batch", "seq")]
    + [(f"split_along_{dim}", {"split_size_or_sections": 2}) for dim in ("batch", "seq")]
    + [("repeat_along_seq", {"repeats": 2})]
    + [(f"index_select_along_{dim}", {"index": torch.tensor([0])}) for dim in ("batch", "seq")]
    + [
        ("permute_along_batch", {"permutation": torch.tensor([0, 1])}),
        ("permute_along_seq", {"permutation": torch.tensor([0, 1, 2])}),
    ]
    + [(f"shuffle_along_{dim}", {}) for dim in ("batch", "seq")]
    + [(f"pad_along_{dim}", {"pad_size": 1}) for dim in ("batch", "seq")]
    + [("clone", {}), ("detach", {}), ("contiguous", {}), ("to", {})]
)


@pytest.mark.parametrize(("name", "kwargs"), CASES, ids=[name for name, _ in CASES])
def test_non_tensor_leaf_raises_type_error(name: str, kwargs: dict[str, Any]) -> None:
    data: Any = {"a": torch.ones(2, 3), "b": "abc"}
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        getattr(nested, name)(data, **kwargs)


@pytest.mark.parametrize(
    "name", ["cat_along_batch", "cat_along_seq", "stack_along_batch", "stack_along_seq"]
)
def test_joining_non_tensor_leaf_raises_type_error(name: str) -> None:
    data: Any = [{"a": "abc"}, {"a": "def"}]
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        getattr(nested, name)(data)
