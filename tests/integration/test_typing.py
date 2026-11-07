r"""Check the static types of the nested functions.

The ``assert_type`` calls are checked by ``ty`` and ``pyright``. At
runtime they do nothing.
"""

from __future__ import annotations

from typing import assert_type

import torch

from batchtensor import nested
from batchtensor.nested import NestedTensor


def test_tensor_in_tensor_out() -> None:
    assert_type(nested.sum_along_batch(torch.ones(2, 3)), torch.Tensor)


def test_dict_in_dict_out() -> None:
    data = {"a": torch.ones(2, 3)}
    assert_type(nested.sum_along_batch(data), dict[str, torch.Tensor])


def test_list_in_list_out() -> None:
    data: list[NestedTensor] = [torch.ones(2, 3)]
    assert_type(nested.slice_along_batch(data, stop=1), list[NestedTensor])


def test_tuple_in_tuple_out() -> None:
    data = (torch.ones(2, 3),)
    assert_type(nested.pad_along_batch(data, 1), tuple[torch.Tensor])
