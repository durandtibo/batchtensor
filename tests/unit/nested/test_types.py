from __future__ import annotations

import torch
from coola.equality import objects_are_equal

from batchtensor.nested import types

##################################
#     Tests for NestedTensor     #
##################################


def test_nested_tensor_is_torch_tensor_union() -> None:
    assert types.NestedTensor is not None


def test_nested_tensor_accepts_tensor() -> None:
    value = torch.tensor([1, 2, 3])
    assert objects_are_equal(value, torch.tensor([1, 2, 3]))


def test_nested_tensor_accepts_mapping() -> None:
    value = {"a": torch.tensor([1, 2, 3])}
    assert objects_are_equal(value, {"a": torch.tensor([1, 2, 3])})


def test_nested_tensor_accepts_nested_mapping() -> None:
    value = {"a": {"b": torch.tensor([1, 2, 3])}}
    assert objects_are_equal(value, {"a": {"b": torch.tensor([1, 2, 3])}})


def test_nested_tensor_accepts_sequence() -> None:
    value = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    assert objects_are_equal(value, [torch.tensor([1, 2]), torch.tensor([3, 4])])
