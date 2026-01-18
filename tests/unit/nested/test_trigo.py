from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola import objects_are_allclose

from batchtensor import nested

if TYPE_CHECKING:
    from collections.abc import Callable

DTYPES = [torch.float, torch.double, torch.long]
POINTWISE_FUNCTIONS = [
    (torch.acos, nested.acos),
    (torch.acosh, nested.acosh),
    (torch.asin, nested.asin),
    (torch.asinh, nested.asinh),
    (torch.atan, nested.atan),
    (torch.atanh, nested.atanh),
    (torch.cos, nested.cos),
    (torch.cosh, nested.cosh),
    (torch.sin, nested.sin),
    (torch.sinh, nested.sinh),
    (torch.tan, nested.tan),
    (torch.tanh, nested.tanh),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("functions", POINTWISE_FUNCTIONS)
def test_pointwise_function_tensor(
    dtype: torch.dtype, functions: tuple[Callable, Callable]
) -> None:
    torch_fn, nested_fn = functions
    tensor = torch.randn(5, 2).to(dtype=dtype)
    assert objects_are_allclose(nested_fn(tensor), torch_fn(tensor), equal_nan=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("functions", POINTWISE_FUNCTIONS)
def test_pointwise_function_dict(dtype: torch.dtype, functions: tuple[Callable, Callable]) -> None:
    torch_fn, nested_fn = functions
    tensor_a = torch.randn(5, 2).to(dtype=dtype)
    tensor_b = torch.randn(5).to(dtype=torch.float)
    tensor_c = torch.randn(5).to(dtype=torch.float)
    assert objects_are_allclose(
        nested_fn(
            {
                "a": tensor_a,
                "b": tensor_b,
                "c": [tensor_c],
            },
        ),
        {
            "a": torch_fn(tensor_a),
            "b": torch_fn(tensor_b),
            "c": [torch_fn(tensor_c)],
        },
        equal_nan=True,
    )
