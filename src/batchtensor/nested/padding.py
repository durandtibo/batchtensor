r"""Implements padding functions for nested data structures."""

from __future__ import annotations

__all__ = ["pad_along_batch", "pad_along_seq", "pad_sequences"]

from functools import partial
from typing import TYPE_CHECKING

from batchtensor import tensor as bt
from batchtensor.utils.structure import apply_to_tensors, combine_tensors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batchtensor.tensor.padding import Side
    from batchtensor.utils.types import T


def pad_along_batch(data: T, pad_size: int, *, value: float = 0.0, side: Side = "right") -> T:
    r"""Pad all the tensors along the batch dimension.

    The padding is added at the end of the batch dimension by default,
    or at the start with ``side="left"``.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors.

    Args:
        data: The input data. Each item must be a tensor.
        pad_size: The number of items to add to the batch dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items. It must be
            representable in the dtype of the tensors.
        side: The side of the dimension where the padding is added,
            ``"right"`` (end) or ``"left"`` (start).

    Returns:
        The padded tensors.

    Raises:
        ValueError: if ``pad_size`` is negative, if ``value`` cannot be
            represented in the dtype of a tensor, or if ``side`` is not
            ``"left"`` or ``"right"``.

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
    return apply_to_tensors(
        data, partial(bt.pad_along_batch, pad_size=pad_size, value=value, side=side)
    )


def pad_along_seq(data: T, pad_size: int, *, value: float = 0.0, side: Side = "right") -> T:
    r"""Pad all the tensors along the sequence dimension.

    The padding is added at the end of the sequence dimension by default,
    or at the start with ``side="left"``.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors.

    Args:
        data: The input data. Each item must be a tensor.
        pad_size: The number of items to add to the sequence dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items. It must be
            representable in the dtype of the tensors.
        side: The side of the dimension where the padding is added,
            ``"right"`` (end) or ``"left"`` (start).

    Returns:
        The padded tensors.

    Raises:
        ValueError: if ``pad_size`` is negative, if ``value`` cannot be
            represented in the dtype of a tensor, or if ``side`` is not
            ``"left"`` or ``"right"``.

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
    return apply_to_tensors(
        data, partial(bt.pad_along_seq, pad_size=pad_size, value=value, side=side)
    )


def pad_sequences(samples: Sequence[T], *, value: float = 0.0, side: Side = "right") -> T:
    r"""Pad the sequences of several samples and stack them into a batch.

    The samples are structures with the same layout. At each leaf position,
    the tensors of the samples are sequences of shape ``(seq_len_i, ...)``
    that can have different lengths along their first dimension. They are
    padded and stacked with ``batchtensor.tensor.pad_sequences``, so the
    leaf of the output has shape ``(len(samples), max_i(seq_len_i), ...)``.
    The container types of the output are the ones of the first sample.

    Note:
        Like ``batchtensor.tensor.pad_sequences``, the dtype and device of
        a leaf of the output are the ones of the leaf of the first sample.
        The tensors of the other samples are copied into it, which casts
        them to this dtype (e.g. a ``float`` value is truncated when the
        first sample has an integer tensor).

    Args:
        samples: The samples to batch. They must have the same keys and
            the same lengths, and each leaf must be a ``torch.Tensor`` with
            at least one dimension. The tensors at a given leaf position
            must have the same shape except for the first dimension.
        value: The fill value used for the padding. It must be
            representable in the dtype of the tensors.
        side: The side of the sequences where the padding is added,
            ``"right"`` (end) or ``"left"`` (start).

    Returns:
        A structure with the layout of the samples. Each leaf is the padded
            tensor of shape ``(batch_size, max_len, ...)``.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``, or if a sample is
            a dataclass.
        ValueError: if ``samples`` is empty, if the samples do not have the
            same keys or lengths, if the trailing shapes of the tensors at
            a leaf position differ, if a tensor is a scalar, if ``value``
            cannot be represented in the dtype of the tensors, or if
            ``side`` is not ``"left"`` or ``"right"``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import pad_sequences
        >>> samples = [
        ...     {"x": torch.tensor([1, 2, 3]), "y": torch.tensor([1.0, 2.0, 3.0])},
        ...     {"x": torch.tensor([4]), "y": torch.tensor([4.0])},
        ... ]
        >>> out = pad_sequences(samples)
        >>> out
        {'x': tensor([[1, 2, 3], [4, 0, 0]]), 'y': tensor([[1., 2., 3.], [4., 0., 0.]])}

        ```

    See Also:
        ``batchtensor.tensor.pad_sequences``: The same function for tensors.
        ``batchtensor.tensor.lengths_to_mask``: Build the mask of a padded batch.
    """
    return combine_tensors(samples, partial(bt.pad_sequences, value=value, side=side))
