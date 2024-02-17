from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from coola import objects_are_equal

from batchtensor.utils.bfs import (
    DefaultTensorIterator,
    IterableTensorIterator,
    IteratorState,
    TensorIterator,
    bfs_tensor,
)

################################
#     Tests for bfs_tensor     #
################################


def test_bfs_tensor_tensor() -> None:
    assert objects_are_equal(list(bfs_tensor(torch.ones(2, 3))), [torch.ones(2, 3)])


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("abc", id="string"),
        pytest.param(42, id="int"),
        pytest.param(4.2, id="float"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param([], id="empty list"),
        pytest.param(("a", "b", "c"), id="tuple"),
        pytest.param((), id="empty tuple"),
        pytest.param({1, 2, 3}, id="set"),
        pytest.param(set(), id="empty set"),
    ],
)
def test_bfs_tensor_no_tensor(data: Any) -> None:
    assert objects_are_equal(list(bfs_tensor(data)), [])


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([torch.ones(2, 3), torch.arange(5)], id="list with only tensors"),
        pytest.param(
            ["abc", torch.ones(2, 3), 42, torch.arange(5)], id="list with non tensor objects"
        ),
        pytest.param((torch.ones(2, 3), torch.arange(5)), id="tuple with only tensors"),
        pytest.param(
            ("abc", torch.ones(2, 3), 42, torch.arange(5)), id="tuple with non tensor objects"
        ),
    ],
)
def test_bfs_tensor_iterable_tensor(data: Any) -> None:
    assert objects_are_equal(list(bfs_tensor(data)), [torch.ones(2, 3), torch.arange(5)])


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({torch.ones(2, 3), torch.arange(5)}, id="set with only tensors"),
        pytest.param(
            {"abc", torch.ones(2, 3), 42, torch.arange(5)}, id="set with non tensor objects"
        ),
    ],
)
def test_bfs_tensor_set(data: Any) -> None:
    assert len(list(bfs_tensor(data))) == 2


def test_bfs_tensor_nested_data() -> None:
    data = [
        torch.ones(2, 3),
        [torch.ones(4), -torch.arange(3), [torch.ones(5)]],
        (1, torch.tensor([42.0]), torch.zeros(2)),
        torch.arange(5),
    ]
    assert objects_are_equal(
        list(bfs_tensor(data)),
        [
            torch.ones(2, 3),
            torch.arange(5),
            torch.ones(4),
            -torch.arange(3),
            torch.tensor([42.0]),
            torch.zeros(2),
            torch.ones(5),
        ],
    )


###########################################
#     Tests for DefaultTensorIterator     #
###########################################


def test_default_tensor_iterator_str() -> None:
    assert str(DefaultTensorIterator()).startswith("DefaultTensorIterator(")


def test_default_tensor_iterator_iterable() -> None:
    state = IteratorState(iterator=TensorIterator(), queue=deque())
    DefaultTensorIterator().iterate("abc", state)
    assert state.queue == deque()


############################################
#     Tests for IterableTensorIterator     #
############################################


def test_iterate_tensor_iterator_str() -> None:
    assert str(IterableTensorIterator()).startswith("IterableTensorIterator(")


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([], id="empty list"),
        pytest.param((), id="empty tuple"),
        pytest.param(set(), id="empty set"),
        pytest.param(deque(), id="empty deque"),
    ],
)
def test_iterate_tensor_iterator_iterate_empty(data: Iterable) -> None:
    state = IteratorState(iterator=TensorIterator(), queue=deque())
    IterableTensorIterator().iterate(data, state)
    assert state.queue == deque()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(["abc", torch.ones(2, 3), 42, torch.arange(5)], id="list"),
        pytest.param(deque(["abc", torch.ones(2, 3), 42, torch.arange(5)]), id="deque"),
        pytest.param(("abc", torch.ones(2, 3), 42, torch.arange(5)), id="tuple"),
    ],
)
def test_iterate_tensor_iterator_iterate(data: Iterable) -> None:
    state = IteratorState(iterator=TensorIterator(), queue=deque())
    IterableTensorIterator().iterate(data, state)
    assert objects_are_equal(list(state.queue), ["abc", torch.ones(2, 3), 42, torch.arange(5)])


####################################
#     Tests for TensorIterator     #
####################################


def test_iterator_str() -> None:
    assert str(TensorIterator()).startswith("TensorIterator(")


@patch.dict(TensorIterator.registry, {}, clear=True)
def test_iterator_add_iterator() -> None:
    iterator = TensorIterator()
    seq_iterator = IterableTensorIterator()
    iterator.add_iterator(list, seq_iterator)
    assert iterator.registry[list] is seq_iterator


@patch.dict(TensorIterator.registry, {}, clear=True)
def test_iterator_add_iterator_duplicate_exist_ok_true() -> None:
    iterator = TensorIterator()
    seq_iterator = IterableTensorIterator()
    iterator.add_iterator(list, DefaultTensorIterator())
    iterator.add_iterator(list, seq_iterator, exist_ok=True)
    assert iterator.registry[list] is seq_iterator


@patch.dict(TensorIterator.registry, {}, clear=True)
def test_iterator_add_iterator_duplicate_exist_ok_false() -> None:
    iterator = TensorIterator()
    seq_iterator = IterableTensorIterator()
    iterator.add_iterator(list, DefaultTensorIterator())
    with pytest.raises(RuntimeError, match="An iterator (.*) is already registered"):
        iterator.add_iterator(list, seq_iterator)


def test_iterator_iterate() -> None:
    state = IteratorState(iterator=TensorIterator(), queue=deque())
    TensorIterator().iterate(["abc", torch.ones(2, 3), 42, torch.arange(5)], state=state)
    assert objects_are_equal(list(state.queue), ["abc", torch.ones(2, 3), 42, torch.arange(5)])


def test_iterator_has_iterator_true() -> None:
    assert TensorIterator().has_iterator(list)


def test_iterator_has_iterator_false() -> None:
    assert not TensorIterator().has_iterator(type(None))


def test_iterator_find_iterator_direct() -> None:
    assert isinstance(TensorIterator().find_iterator(list), IterableTensorIterator)


def test_iterator_find_iterator_indirect() -> None:
    assert isinstance(TensorIterator().find_iterator(str), DefaultTensorIterator)


def test_iterator_find_iterator_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        TensorIterator().find_iterator(Mock(__mro__=[]))


def test_iterator_registry_default() -> None:
    assert len(TensorIterator.registry) >= 7
    assert isinstance(TensorIterator.registry[Iterable], IterableTensorIterator)
    assert isinstance(TensorIterator.registry[deque], IterableTensorIterator)
    assert isinstance(TensorIterator.registry[list], IterableTensorIterator)
    assert isinstance(TensorIterator.registry[object], DefaultTensorIterator)
    assert isinstance(TensorIterator.registry[set], IterableTensorIterator)
    assert isinstance(TensorIterator.registry[str], DefaultTensorIterator)
    assert isinstance(TensorIterator.registry[tuple], IterableTensorIterator)
