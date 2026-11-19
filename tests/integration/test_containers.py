r"""Gates for the container rule (D1) and the empty-structure rule of
``design_principles.md``."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, NamedTuple

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor import nested


class Point(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


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


#######################################
#     Tests for the container kind    #
#######################################

JOIN_FUNCTIONS = [nested.cat_along_batch, nested.stack_along_batch, nested.pad_sequences]


def make_leaf() -> torch.Tensor:
    r"""Return a leaf that every function of ``JOIN_FUNCTIONS`` accepts."""
    return torch.ones(2)


@pytest.mark.parametrize("fn", JOIN_FUNCTIONS)
def test_join_list_and_tuple_returns_a_list(fn: Any) -> None:
    assert type(fn([[make_leaf()], (make_leaf(),)])) is list


@pytest.mark.parametrize("fn", JOIN_FUNCTIONS)
def test_join_tuple_and_list_returns_a_tuple(fn: Any) -> None:
    assert type(fn([(make_leaf(),), [make_leaf()]])) is tuple


@pytest.mark.parametrize("fn", JOIN_FUNCTIONS)
def test_join_dict_and_ordered_dict_returns_a_dict(fn: Any) -> None:
    out = fn([{"a": make_leaf()}, OrderedDict(a=make_leaf())])
    assert type(out) is dict


@pytest.mark.parametrize("fn", JOIN_FUNCTIONS)
def test_join_ordered_dict_and_dict_returns_an_ordered_dict(fn: Any) -> None:
    out = fn([OrderedDict(a=make_leaf()), {"a": make_leaf()}])
    assert type(out) is OrderedDict


@pytest.mark.parametrize("fn", JOIN_FUNCTIONS)
def test_join_mapping_and_sequence_raises(fn: Any) -> None:
    with pytest.raises(ValueError, match="must have the same kind"):
        fn([{"a": make_leaf()}, [make_leaf()]])


@pytest.mark.parametrize("fn", JOIN_FUNCTIONS)
def test_join_sequence_and_mapping_raises(fn: Any) -> None:
    with pytest.raises(ValueError, match="must have the same kind"):
        fn([[make_leaf()], {"a": make_leaf()}])
