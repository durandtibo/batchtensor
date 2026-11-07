from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor import tensor

if TYPE_CHECKING:
    from collections.abc import Callable

# (batchtensor function, torch reference) for the functions that are one torch
# call with the dimension fixed.
CASES = [
    ("sum_along_batch", lambda x: torch.sum(x, dim=0)),
    ("sum_along_seq", lambda x: torch.sum(x, dim=1)),
    ("mean_along_batch", lambda x: torch.mean(x, dim=0)),
    ("prod_along_batch", lambda x: torch.prod(x, dim=0)),
    ("prod_along_seq", lambda x: torch.prod(x, dim=1)),
    ("amax_along_batch", lambda x: torch.amax(x, dim=0)),
    ("amin_along_seq", lambda x: torch.amin(x, dim=1)),
    ("argmax_along_batch", lambda x: torch.argmax(x, dim=0)),
    ("argmin_along_seq", lambda x: torch.argmin(x, dim=1)),
    ("cumsum_along_batch", lambda x: torch.cumsum(x, dim=0)),
    ("cumsum_along_seq", lambda x: torch.cumsum(x, dim=1)),
    ("cumprod_along_batch", lambda x: torch.cumprod(x, dim=0)),
    ("cumprod_along_seq", lambda x: torch.cumprod(x, dim=1)),
    ("argsort_along_batch", lambda x: torch.argsort(x, dim=0)),
    ("argsort_along_seq", lambda x: torch.argsort(x, dim=1)),
    ("select_along_batch", lambda x: torch.select(x, 0, 1)),
    ("select_along_seq", lambda x: torch.select(x, 1, 1)),
]
DTYPES = [torch.float32, torch.float64, torch.int64]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("name", "reference"), CASES, ids=[name for name, _ in CASES])
def test_matches_torch(
    name: str, reference: Callable[..., torch.Tensor], dtype: torch.dtype
) -> None:
    x = (torch.arange(24).reshape(4, 3, 2) % 5 + 1).to(dtype)
    kwargs = {"index": 1} if name.startswith("select") else {}
    if name.startswith("mean") and dtype == torch.int64:
        pytest.skip("torch.mean does not support integer tensors")
    assert objects_are_equal(getattr(tensor, name)(x, **kwargs), reference(x))
