r"""Implements tensor joining functions for nested data structures."""

from __future__ import annotations

__all__ = [
    "cat_along_batch",
    "cat_along_seq",
    "repeat_along_seq",
    "stack_along_batch",
    "stack_along_seq",
]

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, cast

from coola.recursive import recursive_apply

from batchtensor import tensor as bt

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    import torch

    from batchtensor.nested.types import NestedTensor


def cat_along_batch(data: Sequence[NestedTensor]) -> NestedTensor:
    r"""Concatenate the given tensors in the batch dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data to concatenate. The dictionaries must have
            the same keys.

    Raises:
        KeyError: if the dictionaries do not all have the same keys.

    Returns:
        The concatenated tensors along the batch dimension.

    Example:
        ```pycon

        >>> import torch
        >>> from batchtensor.nested import cat_along_batch
        >>> data = [
        ...     {
        ...         "a": torch.tensor([[0, 1, 2], [4, 5, 6]]),
        ...         "b": torch.tensor([[10, 11, 12], [13, 14, 15]]),
        ...     },
        ...     {"a": torch.tensor([[7, 8, 9]]), "b": torch.tensor([[17, 18, 19]])},
        ... ]
        >>> out = cat_along_batch(data)
        >>> out
        {'a': tensor([[0, 1, 2], [4, 5, 6], [7, 8, 9]]),
         'b': tensor([[10, 11, 12], [13, 14, 15], [17, 18, 19]])}

        ```
    """
    return _combine(data, bt.cat_along_batch)


def cat_along_seq(data: Sequence[NestedTensor]) -> NestedTensor:
    r"""Concatenate the given tensors in the sequence dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data to concatenate. The dictionaries must have
            the same keys.

    Raises:
        KeyError: if the dictionaries do not all have the same keys.

    Returns:
        The concatenated tensors along the sequence dimension.

    Example:
        ```pycon

        >>> import torch
        >>> from batchtensor.nested import cat_along_seq
        >>> data = [
        ...     {
        ...         "a": torch.tensor([[0, 1, 2], [4, 5, 6]]),
        ...         "b": torch.tensor([[10, 11, 12], [13, 14, 15]]),
        ...     },
        ...     {"a": torch.tensor([[7], [8]]), "b": torch.tensor([[17], [18]])},
        ... ]
        >>> out = cat_along_seq(data)
        >>> out
        {'a': tensor([[0, 1, 2, 7], [4, 5, 6, 8]]),
         'b': tensor([[10, 11, 12, 17], [13, 14, 15, 18]])}

        ```
    """
    return _combine(data, bt.cat_along_seq)


def repeat_along_seq(data: NestedTensor, repeats: int) -> NestedTensor:
    r"""Repeat all the tensors along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        repeats: The number of times to repeat
            the data along the sequence dimension.

    Returns:
        The tensors repeated along the sequence dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import repeat_along_seq
        >>> data = {
        ...     "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": torch.tensor([[4, 3, 2, 1, 0]]),
        ... }
        >>> out = repeat_along_seq(data, 2)
        >>> out
        {'a': tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
         'b': tensor([[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]])}

        ```
    """
    return recursive_apply(data, partial(bt.repeat_along_seq, repeats=repeats))


def stack_along_batch(data: Sequence[NestedTensor]) -> NestedTensor:
    r"""Stack the given samples along a new batch dimension.

    This is the nested equivalent of a collate function: given a
    sequence of samples (dictionaries of tensors without a batch
    dimension), it combines them into a single batched dictionary. Each
    tensor must have the same shape across all the samples.

    Args:
        data: The input samples to stack. The dictionaries must have
            the same keys.

    Raises:
        KeyError: if the dictionaries do not all have the same keys.

    Returns:
        The stacked tensors along a new batch dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import stack_along_batch
        >>> data = [
        ...     {"a": torch.tensor([0, 1, 2]), "b": torch.tensor(1)},
        ...     {"a": torch.tensor([3, 4, 5]), "b": torch.tensor(2)},
        ... ]
        >>> out = stack_along_batch(data)
        >>> out
        {'a': tensor([[0, 1, 2], [3, 4, 5]]), 'b': tensor([1, 2])}

        ```

    See Also:
        ``cat_along_batch``: Concatenate along an existing batch dimension.
        ``stack_along_seq``: Stack along a new sequence dimension instead.
    """
    return _combine(data, bt.stack_along_batch)


def stack_along_seq(data: Sequence[NestedTensor]) -> NestedTensor:
    r"""Stack the given samples along a new sequence dimension.

    Combines a sequence of per-step samples (dictionaries of tensors
    without a sequence dimension) into a single dictionary with a
    sequence dimension. Each tensor must have the same shape across all
    the samples.

    Args:
        data: The input samples to stack. The dictionaries must have
            the same keys.

    Raises:
        KeyError: if the dictionaries do not all have the same keys.

    Returns:
        The stacked tensors along a new sequence dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import stack_along_seq
        >>> data = [
        ...     {"a": torch.tensor([0, 1]), "b": torch.tensor([10, 11])},
        ...     {"a": torch.tensor([2, 3]), "b": torch.tensor([12, 13])},
        ... ]
        >>> out = stack_along_seq(data)
        >>> out
        {'a': tensor([[0, 2], [1, 3]]), 'b': tensor([[10, 12], [11, 13]])}

        ```

    See Also:
        ``cat_along_seq``: Concatenate along an existing sequence dimension.
        ``stack_along_batch``: Stack along a new batch dimension instead.
    """
    return _combine(data, bt.stack_along_seq)


def _combine(
    data: Sequence[NestedTensor], fn: Callable[[list[torch.Tensor]], torch.Tensor]
) -> NestedTensor:
    r"""Combine a sequence of nested structures with the same structure.

    Dictionaries, lists and tuples are traversed recursively and ``fn`` is
    applied to the list of tensors found at each leaf position. An empty
    sequence returns an empty dictionary.

    Raises:
        KeyError: if the dictionaries do not have the same keys.
        ValueError: if the lists/tuples do not have the same length, or if
            the structures do not match.
    """
    if not data:
        return {}
    first = data[0]
    if isinstance(first, Mapping):
        keys = set(first)
        for i, item in enumerate(data[1:], start=1):
            if not isinstance(item, Mapping) or set(item) != keys:
                other = list(item) if isinstance(item, Mapping) else type(item).__name__
                msg = (
                    "the dictionaries must have the same keys but item 0 has keys "
                    f"{list(keys)} and item {i} has keys {other}"
                )
                raise KeyError(msg)
        maps = cast("Sequence[Mapping[Hashable, NestedTensor]]", data)
        return {key: _combine([d[key] for d in maps], fn) for key in first}
    if isinstance(first, (list, tuple)):
        for i, item in enumerate(data[1:], start=1):
            if not isinstance(item, (list, tuple)) or len(item) != len(first):
                msg = f"the sequences must have the same length but item {i} does not match item 0"
                raise ValueError(msg)
        out = [_combine([d[j] for d in data], fn) for j in range(len(first))]
        return out if isinstance(first, list) else tuple(out)
    return fn(cast("list[torch.Tensor]", list(data)))
