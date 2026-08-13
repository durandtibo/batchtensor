r"""Implements padding functions for nested data structures."""

from __future__ import annotations

__all__ = ["pad_along_batch", "pad_along_seq"]

from functools import partial
from typing import TYPE_CHECKING

from coola.recursive import recursive_apply

from batchtensor import tensor as bt

if TYPE_CHECKING:
    from batchtensor.nested.types import NestedTensor


def pad_along_batch(data: NestedTensor, pad_size: int, value: float = 0.0) -> NestedTensor:
    r"""Pad all the tensors along the batch dimension.

    The padding is added at the end of the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors.

    Args:
        data: The input data. Each item must be a tensor.
        pad_size: The number of items to add to the batch dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items.

    Returns:
        The padded tensors.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import pad_along_batch
        >>> data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
        >>> out = pad_along_batch(data, pad_size=1)
        >>> out
        {'a': tensor([[0, 1], [2, 3], [0, 0]]), 'b': tensor([4, 5, 0])}

        ```

    See Also:
        ``pad_along_seq``: Pad along the sequence dimension instead.
    """
    return recursive_apply(data, partial(bt.pad_along_batch, pad_size=pad_size, value=value))


def pad_along_seq(data: NestedTensor, pad_size: int, value: float = 0.0) -> NestedTensor:
    r"""Pad all the tensors along the sequence dimension.

    The padding is added at the end of the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors.

    Args:
        data: The input data. Each item must be a tensor.
        pad_size: The number of items to add to the sequence dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items.

    Returns:
        The padded tensors.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import pad_along_seq
        >>> data = {
        ...     "a": torch.tensor([[0, 1, 2], [3, 4, 5]]),
        ...     "b": torch.tensor([[6, 7, 8]]),
        ... }
        >>> out = pad_along_seq(data, pad_size=1)
        >>> out
        {'a': tensor([[0, 1, 2, 0], [3, 4, 5, 0]]), 'b': tensor([[6, 7, 8, 0]])}

        ```

    See Also:
        ``pad_along_batch``: Pad along the batch dimension instead.
        ``lengths_to_mask``: Build a mask that flags the padded positions.
    """
    return recursive_apply(data, partial(bt.pad_along_seq, pad_size=pad_size, value=value))
