r"""Defines type aliases used by the ``batchtensor.nested`` package."""

from __future__ import annotations

__all__ = ["NestedTensor"]

from collections.abc import Hashable, Mapping, Sequence

import torch

# A recursive alias for the nested data structures accepted by the functions
# in ``batchtensor.nested``: a tensor, or an arbitrarily nested mapping /
# sequence of such structures (e.g. a dict of lists of tensors).
NestedTensor = torch.Tensor | Mapping[Hashable, "NestedTensor"] | Sequence["NestedTensor"]
