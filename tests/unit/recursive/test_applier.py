from __future__ import annotations

from collections.abc import Mapping, Sequence
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal

from batchtensor.recursive import (
    Applier,
    ApplyState,
    DefaultApplier,
    MappingApplier,
    SequenceApplier,
)


@pytest.fixture()
def state() -> ApplyState:
    return ApplyState(applier=Applier())


#############################
#     Tests for Applier     #
#############################


def test_applier_str() -> None:
    assert str(Applier()).startswith("Applier(")


@patch.dict(Applier.registry, {}, clear=True)
def test_applier_add_applier() -> None:
    applier = Applier()
    seq_applier = SequenceApplier()
    applier.add_applier(list, seq_applier)
    assert applier.registry[list] is seq_applier


@patch.dict(Applier.registry, {}, clear=True)
def test_applier_add_applier_duplicate_exist_ok_true() -> None:
    applier = Applier()
    seq_applier = SequenceApplier()
    applier.add_applier(list, MappingApplier())
    applier.add_applier(list, seq_applier, exist_ok=True)
    assert applier.registry[list] == seq_applier


@patch.dict(Applier.registry, {}, clear=True)
def test_applier_add_applier_duplicate_exist_ok_false() -> None:
    applier = Applier()
    seq_applier = SequenceApplier()
    applier.add_applier(list, MappingApplier())
    with pytest.raises(RuntimeError, match="An applier (.*) is already registered"):
        applier.add_applier(list, seq_applier)


def test_applier_apply(state: ApplyState) -> None:
    assert objects_are_equal(Applier().apply([1, "abc"], str, state=state), ["1", "abc"])


def test_applier_apply_nested(state: ApplyState) -> None:
    assert objects_are_equal(
        Applier().apply(
            {"list": [1, "abc"], "set": {1, 2, 3}, "dict": {"a": 1, "b": "abc"}}, str, state=state
        ),
        {"list": ["1", "abc"], "set": {"1", "2", "3"}, "dict": {"a": "1", "b": "abc"}},
    )


def test_applier_has_applier_true() -> None:
    assert Applier().has_applier(dict)


def test_applier_has_applier_false() -> None:
    assert not Applier().has_applier(type(None))


def test_applier_find_applier_direct() -> None:
    assert isinstance(Applier().find_applier(dict), MappingApplier)


def test_applier_find_applier_indirect() -> None:
    assert isinstance(Applier().find_applier(str), DefaultApplier)


def test_applier_find_applier_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        Applier().find_applier(Mock(__mro__=[]))


def test_applier_registry_default() -> None:
    assert len(Applier.registry) >= 7
    assert isinstance(Applier.registry[Mapping], MappingApplier)
    assert isinstance(Applier.registry[Sequence], SequenceApplier)
    assert isinstance(Applier.registry[dict], MappingApplier)
    assert isinstance(Applier.registry[list], SequenceApplier)
    assert isinstance(Applier.registry[object], DefaultApplier)
    assert isinstance(Applier.registry[set], SequenceApplier)
    assert isinstance(Applier.registry[tuple], SequenceApplier)
