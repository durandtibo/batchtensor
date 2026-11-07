from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from batchtensor import nested, tensor

if TYPE_CHECKING:
    from collections.abc import Callable

OPTIONS = (
    "keepdim",
    "descending",
    "stable",
    "step",
    "value",
    "generator",
    "max_len",
    "memory_format",
)

TENSOR_FUNCTIONS = [(name, getattr(tensor, name)) for name in tensor.__all__]
NESTED_FUNCTIONS = [
    (name, getattr(nested, name)) for name in nested.__all__ if name != "NestedTensor"
]

######################################
#     Tests for keyword-only options #
######################################


def get_positional_options(fn: Callable[..., object]) -> list[str]:
    return [
        name
        for name, param in inspect.signature(fn).parameters.items()
        if name in OPTIONS and param.kind is not inspect.Parameter.KEYWORD_ONLY
    ]


@pytest.mark.parametrize("fn", [f for _, f in TENSOR_FUNCTIONS])
def test_tensor_options_are_keyword_only(fn: Callable[..., object]) -> None:
    assert get_positional_options(fn) == []


@pytest.mark.parametrize("fn", [f for _, f in NESTED_FUNCTIONS])
def test_nested_options_are_keyword_only(fn: Callable[..., object]) -> None:
    assert get_positional_options(fn) == []


@pytest.mark.parametrize(("name", "fn"), TENSOR_FUNCTIONS + NESTED_FUNCTIONS)
def test_no_var_keyword_except_to(name: str, fn: Callable[..., object]) -> None:
    has_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(fn).parameters.values()
    )
    assert has_var_keyword == (name == "to")
