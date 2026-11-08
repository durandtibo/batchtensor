r"""Implements functions to read the size of nested data structures."""

from __future__ import annotations

__all__ = ["batch_size", "seq_len"]

from typing import Any

from batchtensor.constants import BATCH_DIM, SEQ_DIM
from batchtensor.utils.structure import get_size


def batch_size(data: Any) -> int:
    r"""Return the batch size of a nested data structure.

    The batch size is the size of the batch dimension (dimension 0) shared
    by all the tensors.

    Args:
        data: The nested data structure. Each leaf must be a
            ``torch.Tensor`` with at least one dimension.

    Returns:
        The size of the batch dimension.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``data`` has no tensor, if a tensor has no batch
            dimension, or if the tensors have different batch sizes.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import batch_size
        >>> batch_size({"a": torch.ones(2, 3), "b": [torch.ones(2)]})
        2

        ```

    See Also:
        ``batchtensor.nested.seq_len``: Return the sequence length.
    """
    return get_size(data, BATCH_DIM)


def seq_len(data: Any) -> int:
    r"""Return the sequence length of a nested data structure.

    The sequence length is the size of the sequence dimension (dimension 1)
    shared by all the tensors.

    Args:
        data: The nested data structure. Each leaf must be a
            ``torch.Tensor`` with at least two dimensions.

    Returns:
        The size of the sequence dimension.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``data`` has no tensor, if a tensor has no sequence
            dimension, or if the tensors have different sequence lengths.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import seq_len
        >>> seq_len({"a": torch.ones(2, 3), "b": [torch.ones(2, 3, 4)]})
        3

        ```

    See Also:
        ``batchtensor.nested.batch_size``: Return the batch size.
    """
    return get_size(data, SEQ_DIM)
