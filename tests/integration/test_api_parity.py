from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from batchtensor import nested, tensor

if TYPE_CHECKING:
    from collections.abc import Callable

# The functions of ``tensor`` that have no ``nested`` twin because their input
# is not a batch of tensors.
TENSOR_ONLY = {"lengths_to_mask", "pad_sequences"}

# The functions of ``nested`` that have no ``tensor`` twin.
NESTED_ONLY = {
    "NestedTensor",
    "clone",
    "contiguous",
    "detach",
    "from_numpy",
    "pin_memory",
    "to",
    "to_numpy",
}

SHARED_NAMES = sorted(set(tensor.__all__) - TENSOR_ONLY)

##########################################
#     Tests for the names of the API     #
##########################################


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_tensor_function_has_nested_twin(name: str) -> None:
    assert name in nested.__all__


def test_nested_only_names() -> None:
    assert sorted(set(nested.__all__) - set(tensor.__all__)) == sorted(NESTED_ONLY)


def test_tensor_only_names() -> None:
    assert sorted(set(tensor.__all__) - set(nested.__all__)) == sorted(TENSOR_ONLY)


#############################################
#     Tests for the signatures of twins     #
#############################################


def get_options(fn: Callable[..., object]) -> list[tuple[str, object, object, object]]:
    r"""Return the name, kind, default and annotation of the parameters
    after the first one (the data)."""
    params = list(inspect.signature(fn).parameters.values())[1:]
    return [(p.name, p.kind, p.default, p.annotation) for p in params]


def get_data_kind(fn: Callable[..., object]) -> object:
    return next(iter(inspect.signature(fn).parameters.values())).kind


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_twins_have_the_same_options(name: str) -> None:
    assert get_options(getattr(nested, name)) == get_options(getattr(tensor, name))


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_twins_have_the_same_data_parameter_kind(name: str) -> None:
    assert get_data_kind(getattr(nested, name)) == get_data_kind(getattr(tensor, name))
