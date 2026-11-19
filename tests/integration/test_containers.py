r"""Gates for the container rule (D1), the empty-structure rule and the
documented limit on dataclasses of ``design_principles.md``."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor import nested


class Point(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class Pair:
    first: torch.Tensor
    second: torch.Tensor


def make_tensor() -> torch.Tensor:
    return torch.arange(8, dtype=torch.float).reshape(4, 2)


def make_default() -> torch.Tensor:
    return torch.zeros(4, 2)


def make_structures() -> dict[str, Any]:
    return {
        "dict": {"a": make_tensor(), "b": make_tensor()},
        "list": [make_tensor(), make_tensor()],
        "tuple": (make_tensor(), make_tensor()),
        "named_tuple": Point(make_tensor(), make_tensor()),
        "ordered_dict": OrderedDict(b=make_tensor(), a=make_tensor()),
        "defaultdict": defaultdict(make_default, {"a": make_tensor()}),
        "nested_mix": {
            "a": [make_tensor(), (make_tensor(), Point(make_tensor(), make_tensor()))],
            "b": OrderedDict(c={"d": make_tensor()}),
        },
    }


STRUCTURES = list(make_structures())


def get_layout(data: Any) -> Any:
    r"""Return the containers of a structure, with a placeholder for a leaf."""
    if isinstance(data, torch.Tensor):
        return "leaf"
    if isinstance(data, dict):
        return (type(data), {key: get_layout(value) for key, value in data.items()})
    return (type(data), [get_layout(value) for value in data])


#######################################################
#     Tests for the container rule (same for all)     #
#######################################################


@pytest.mark.parametrize("name", STRUCTURES)
def test_map_keeps_the_container_types(name: str) -> None:
    data = make_structures()[name]
    assert get_layout(nested.sum_along_batch(data)) == get_layout(data)


@pytest.mark.parametrize("name", STRUCTURES)
def test_split_keeps_the_container_types(name: str) -> None:
    data = make_structures()[name]
    assert [get_layout(part) for part in nested.chunk_along_batch(data, 2)] == [
        get_layout(data)
    ] * 2


@pytest.mark.parametrize("name", STRUCTURES)
def test_combine_keeps_the_container_types(name: str) -> None:
    data = make_structures()[name]
    assert get_layout(nested.cat_along_batch([data, make_structures()[name]])) == get_layout(data)


@pytest.mark.parametrize("name", STRUCTURES)
def test_cat_is_the_inverse_of_chunk(name: str) -> None:
    data = make_structures()[name]
    assert objects_are_equal(nested.cat_along_batch(nested.chunk_along_batch(data, 2)), data)


######################################
#     Tests for empty structures     #
######################################

EMPTY = [{}, [], ()]


@pytest.mark.parametrize("data", EMPTY)
def test_map_keeps_the_empty_structure(data: Any) -> None:
    assert objects_are_equal(nested.sum_along_batch(data), data)


@pytest.mark.parametrize("data", EMPTY)
def test_chunk_raises_on_the_empty_structure(data: Any) -> None:
    with pytest.raises(ValueError, match=r"Cannot split data that does not contain any tensor"):
        nested.chunk_along_batch(data, 2)


@pytest.mark.parametrize("data", EMPTY)
def test_split_raises_on_the_empty_structure(data: Any) -> None:
    with pytest.raises(ValueError, match=r"Cannot split data that does not contain any tensor"):
        nested.split_along_batch(data, 2)


@pytest.mark.parametrize("data", EMPTY)
def test_shuffle_raises_on_the_empty_structure(data: Any) -> None:
    with pytest.raises(
        ValueError, match=r"Cannot get the size of data that does not contain any tensor"
    ):
        nested.shuffle_along_batch(data)


##########################################
#     Tests for the documented limit     #
##########################################


def test_cat_raises_on_a_dataclass() -> None:
    items: Any = [Pair(make_tensor(), make_tensor()), Pair(make_tensor(), make_tensor())]
    with pytest.raises(
        TypeError, match=r"dataclasses are not supported by combine_tensors; got Pair"
    ):
        nested.cat_along_batch(items)
