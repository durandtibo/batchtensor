r"""Defines the type aliases of nested structures of tensors.

They live in ``batchtensor.utils`` because the helpers of
``batchtensor.utils.structure`` use them. ``NestedTensor`` is re-exported by
``batchtensor.nested``."""

from __future__ import annotations

__all__ = ["NestedTensor", "T"]

from collections.abc import Mapping
from typing import Any, TypeVar

import torch

# A recursive alias for the nested data structures accepted by the functions
# in ``batchtensor.nested`` and by the helpers of ``batchtensor.utils.structure``: a tensor, or an arbitrarily nested mapping, list
# or tuple of such structures (e.g. a dict of lists of tensors). ``str`` and
# ``bytes`` are not accepted.
NestedTensor = (
    torch.Tensor | Mapping[Any, "NestedTensor"] | list["NestedTensor"] | tuple["NestedTensor", ...]
)

# The type of the input of a function that keeps the structure of its input.
T = TypeVar("T", bound=NestedTensor)
