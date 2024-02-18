from collections.abc import Callable

import pytest
import torch
from coola import objects_are_allclose

from batchtensor import nested

DTYPES = [torch.float, torch.double, torch.long]
POINTWISE_FUNCTIONS = [
    (torch.acosh, nested.acosh),
    (torch.asinh, nested.asinh),
    (torch.atanh, nested.atanh),
    (torch.acos, nested.acos),
    (torch.asin, nested.asin),
    (torch.atan, nested.atan),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("functions", POINTWISE_FUNCTIONS)
def test_pointwise_function_tensor(
    dtype: torch.dtype, functions: tuple[Callable, Callable]
) -> None:
    torch_fn, nested_fn = functions
    tensor = torch.randn(5, 2).to(dtype=dtype)
    assert objects_are_allclose(nested_fn(tensor), torch_fn(tensor), equal_nan=True)
