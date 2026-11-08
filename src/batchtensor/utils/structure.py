r"""Implement the helpers that traverse nested structures of tensors.

All the functions of ``batchtensor.nested`` go through the helpers of this
module, so the rules for containers and leaves are defined in one place:

- the container types, keys and order are handled by ``coola.recursive``;
- a leaf that is not a ``torch.Tensor`` raises a ``TypeError``.
"""

from __future__ import annotations

__all__ = ["apply_to_tensors", "combine_tensors", "get_size", "split_tensors"]

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, final

import torch
from coola.iterator import dfs_iterate
from coola.recursive import recursive_apply
from coola.recursive.rebuild import rebuild_mapping, rebuild_sequence

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batchtensor.utils.types import T


def apply_to_tensors(data: T, fn: Callable[[torch.Tensor], Any]) -> T:
    r"""Apply a function to every tensor of a nested structure.

    The structure of ``data`` is kept: the container types, the keys and the
    order are the ones of ``coola.recursive.recursive_apply``, and every
    leaf is replaced by the output of ``fn``.

    Args:
        data: The nested structure. Each leaf must be a ``torch.Tensor``.
        fn: The function applied to each tensor.

    Returns:
        A structure with the same containers as ``data`` and the outputs
            of ``fn`` as leaves.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.utils.structure import apply_to_tensors
        >>> out = apply_to_tensors({"a": torch.zeros(2, 3), "b": [torch.ones(2)]}, torch.exp)
        >>> out
        {'a': tensor([[1., 1., 1.], [1., 1., 1.]]), 'b': [tensor([2.7183, 2.7183])]}

        ```
    """
    return recursive_apply(data, _guard(fn))


def split_tensors(data: T, fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]]) -> tuple[T, ...]:
    r"""Split every tensor of a nested structure and regroup the pieces.

    The i-th output has the same structure as ``data`` and contains the
    i-th piece of every tensor.

    Args:
        data: The nested structure to split. Each leaf must be a
            ``torch.Tensor``.
        fn: The function that splits a tensor into a tuple of tensors,
            e.g. ``lambda t: t.chunk(2, dim=0)``.

    Returns:
        A tuple of nested structures with the same structure as ``data``.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``data`` has no tensor, or if the tensors are not
            split into the same, non-zero, number of pieces.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.utils.structure import split_tensors
        >>> data = {"a": torch.arange(4), "b": [torch.arange(4)]}
        >>> out = split_tensors(data, lambda t: t.chunk(2, dim=0))
        >>> out
        ({'a': tensor([0, 1]), 'b': [tensor([0, 1])]}, {'a': tensor([2, 3]), 'b': [tensor([2, 3])]})

        ```
    """
    parts = apply_to_tensors(data, lambda tensor: _Parts(tuple(fn(tensor))))
    counts = {leaf.parts.__len__() for leaf in dfs_iterate(parts)}
    if not counts:
        msg = "Cannot split data that does not contain any tensor"
        raise ValueError(msg)
    if len(counts) > 1:
        msg = (
            f"All the tensors must be split into the same number of pieces but got {sorted(counts)}"
        )
        raise ValueError(msg)
    num_parts = counts.pop()
    if num_parts == 0:
        msg = "The tensors must be split into at least one piece"
        raise ValueError(msg)
    return tuple(recursive_apply(parts, lambda leaf, i=i: leaf.parts[i]) for i in range(num_parts))


def combine_tensors(items: Sequence[T], fn: Callable[[list[torch.Tensor]], torch.Tensor]) -> T:
    r"""Combine structures with the same layout into one structure.

    The structures are traversed in parallel and ``fn`` is applied to the
    list of tensors found at each leaf position. The container types of the
    output are the ones of the first item. Dataclasses are not supported.

    Args:
        items: The structures to combine. They must have the same keys
            and the same lengths.
        fn: The function that combines a list of tensors into one tensor,
            e.g. ``lambda tensors: torch.cat(tensors, dim=0)``.

    Returns:
        A structure with the same layout as the items.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``, or if an item is a
            dataclass.
        ValueError: if ``items`` is empty, or if the structures do not have
            the same keys or the same lengths.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.utils.structure import combine_tensors
        >>> items = [{"a": torch.ones(2, 3)}, {"a": torch.zeros(1, 3)}]
        >>> out = combine_tensors(items, lambda tensors: torch.cat(tensors, dim=0))
        >>> out["a"].shape
        torch.Size([3, 3])

        ```
    """
    return _combine_tensors(items, fn)


def _combine_tensors(items: Sequence[Any], fn: Callable[[list[torch.Tensor]], torch.Tensor]) -> Any:
    r"""Combine structures with the same layout into one structure.

    This is the untyped implementation of ``combine_tensors``: the recursion
    goes through containers whose type is not known statically, so the
    public function declares the type.

    Args:
        items: The structures to combine.
        fn: The function that combines a list of tensors into one tensor.

    Returns:
        A structure with the same layout as the items.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``, or if an item is a
            dataclass.
        ValueError: if ``items`` is empty, or if the structures do not have
            the same keys or the same lengths.
    """
    if not items:
        msg = "Cannot combine an empty sequence of structures"
        raise ValueError(msg)
    first = items[0]
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        msg = f"dataclasses are not supported by combine_tensors; got {type(first).__qualname__}"
        raise TypeError(msg)
    if isinstance(first, Mapping):
        keys = set(first)
        for i, item in enumerate(items[1:], start=1):
            if not isinstance(item, Mapping) or set(item) != keys:
                other = list(item) if isinstance(item, Mapping) else type(item).__qualname__
                msg = (
                    "The mappings must have the same keys but item 0 has keys "
                    f"{list(first)} and item {i} has keys {other}"
                )
                raise ValueError(msg)
        return rebuild_mapping(
            first, {key: _combine_tensors([item[key] for item in items], fn) for key in first}
        )
    if isinstance(first, (list, tuple)):
        for i, item in enumerate(items[1:], start=1):
            if not isinstance(item, (list, tuple)) or len(item) != len(first):
                msg = f"The sequences must have the same length but item {i} does not match item 0"
                raise ValueError(msg)
        return rebuild_sequence(
            first,
            [_combine_tensors([item[j] for item in items], fn) for j in range(len(first))],
        )
    for item in items:
        _check_tensor(item)
    return fn(list(items))


def get_size(data: Any, dim: int) -> int:
    r"""Get the size of a dimension shared by all the tensors.

    Args:
        data: The nested structure. Each leaf must be a ``torch.Tensor``.
        dim: The dimension.

    Returns:
        The size of the dimension ``dim`` of the tensors.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``data`` has no tensor, if a tensor has no dimension
            ``dim``, or if the tensors have different sizes along ``dim``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.utils.structure import get_size
        >>> get_size({"a": torch.ones(2, 3), "b": [torch.ones(2)]}, dim=0)
        2

        ```
    """
    sizes = set()
    for leaf in dfs_iterate(data):
        _check_tensor(leaf)
        if leaf.ndim <= dim:
            msg = f"Expected a tensor with at least {dim + 1} dimension(s) but got shape {tuple(leaf.shape)}"
            raise ValueError(msg)
        sizes.add(leaf.shape[dim])
    if not sizes:
        msg = "Cannot get the size of data that does not contain any tensor"
        raise ValueError(msg)
    if len(sizes) > 1:
        msg = (
            f"All the tensors must have the same size along dimension {dim} but got {sorted(sizes)}"
        )
        raise ValueError(msg)
    return sizes.pop()


@final
class _Parts:
    r"""Wrap the pieces of a split tensor so that they are one leaf."""

    __slots__ = ("parts",)

    def __init__(self, parts: tuple[torch.Tensor, ...]) -> None:
        self.parts = parts


def _check_tensor(leaf: Any) -> None:
    r"""Check that a leaf is a tensor.

    Args:
        leaf: The leaf to check.

    Raises:
        TypeError: if the leaf is not a ``torch.Tensor``.
    """
    if not isinstance(leaf, torch.Tensor):
        msg = f"Expected a torch.Tensor leaf but received {type(leaf).__qualname__}"
        raise TypeError(msg)


def _guard(fn: Callable[[torch.Tensor], Any]) -> Callable[[Any], Any]:
    r"""Wrap a function so that it only accepts tensors.

    Args:
        fn: The function to wrap.

    Returns:
        A function that calls ``fn`` on a tensor and raises a
            ``TypeError`` on any other input.
    """

    def guarded(leaf: Any) -> Any:
        _check_tensor(leaf)
        return fn(leaf)

    return guarded
