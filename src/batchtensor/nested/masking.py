r"""Implements masked reduction functions for nested data structures."""

from __future__ import annotations

__all__ = [
    "masked_amax_along_seq",
    "masked_amin_along_seq",
    "masked_mean_along_seq",
    "masked_sum_along_seq",
]

from functools import partial
from typing import TYPE_CHECKING

from batchtensor import tensor as bt
from batchtensor.utils.structure import apply_to_tensors

if TYPE_CHECKING:
    import torch

    from batchtensor.utils.types import T


def masked_sum_along_seq(data: T, mask: torch.Tensor, *, keepdim: bool = False) -> T:
    r"""Return the sum of the valid steps of every tensor along the sequence dimension.

    ``mask`` is one tensor shared by all the leaves. See
    ``batchtensor.tensor.masked_sum_along_seq`` for the semantics.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        data: The input data. Each leaf must be a ``torch.Tensor`` of shape
            ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is shared by all the leaves.
        keepdim: Whether the output tensors have dim retained or not.

    Returns:
        A structure with the same layout as ``data``. Each leaf is the sum of the valid steps.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``mask`` is not a boolean tensor, if it is not 2-D,
            or if its shape is not the two first dimensions of a leaf.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import masked_sum_along_seq
        >>> from batchtensor.tensor import lengths_to_mask
        >>> data = {"a": torch.tensor([[1, 2, 3], [4, 5, 6]]), "b": [torch.tensor([[1, 2, 3], [4, 5, 6]])]}
        >>> mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
        >>> out = masked_sum_along_seq(data, mask)
        >>> out
        {'a': tensor([3, 4]), 'b': [tensor([3, 4])]}

        ```

    See Also:
        ``batchtensor.tensor.masked_sum_along_seq``: The same function for tensors.
    """
    return apply_to_tensors(data, partial(bt.masked_sum_along_seq, mask=mask, keepdim=keepdim))


def masked_mean_along_seq(data: T, mask: torch.Tensor, *, keepdim: bool = False) -> T:
    r"""Return the mean of the valid steps of every tensor along the sequence dimension.

    ``mask`` is one tensor shared by all the leaves. See
    ``batchtensor.tensor.masked_mean_along_seq`` for the semantics.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        data: The input data. Each leaf must be a ``torch.Tensor`` of shape
            ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is shared by all the leaves.
        keepdim: Whether the output tensors have dim retained or not.

    Returns:
        A structure with the same layout as ``data``. Each leaf is the mean of the valid steps (``nan`` for a row without valid step).

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if a leaf does not have a floating point or complex dtype, if ``mask`` is not a boolean tensor, if it is not 2-D,
            or if its shape is not the two first dimensions of a leaf.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import masked_mean_along_seq
        >>> from batchtensor.tensor import lengths_to_mask
        >>> data = {"a": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "b": [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]}
        >>> mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
        >>> out = masked_mean_along_seq(data, mask)
        >>> out
        {'a': tensor([1.5000, 4.0000]), 'b': [tensor([1.5000, 4.0000])]}

        ```

    See Also:
        ``batchtensor.tensor.masked_mean_along_seq``: The same function for tensors.
    """
    return apply_to_tensors(data, partial(bt.masked_mean_along_seq, mask=mask, keepdim=keepdim))


def masked_amax_along_seq(data: T, mask: torch.Tensor, *, keepdim: bool = False) -> T:
    r"""Return the maximum of the valid steps of every tensor along the sequence dimension.

    ``mask`` is one tensor shared by all the leaves. See
    ``batchtensor.tensor.masked_amax_along_seq`` for the semantics.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        data: The input data. Each leaf must be a ``torch.Tensor`` of shape
            ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is shared by all the leaves.
        keepdim: Whether the output tensors have dim retained or not.

    Returns:
        A structure with the same layout as ``data``. Each leaf is the maximum of the valid steps.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``mask`` is not a boolean tensor, if it is not 2-D,
            or if its shape is not the two first dimensions of a leaf.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import masked_amax_along_seq
        >>> from batchtensor.tensor import lengths_to_mask
        >>> data = {"a": torch.tensor([[1, 2, 3], [4, 5, 6]]), "b": [torch.tensor([[1, 2, 3], [4, 5, 6]])]}
        >>> mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
        >>> out = masked_amax_along_seq(data, mask)
        >>> out
        {'a': tensor([2, 4]), 'b': [tensor([2, 4])]}

        ```

    See Also:
        ``batchtensor.tensor.masked_amax_along_seq``: The same function for tensors.
    """
    return apply_to_tensors(data, partial(bt.masked_amax_along_seq, mask=mask, keepdim=keepdim))


def masked_amin_along_seq(data: T, mask: torch.Tensor, *, keepdim: bool = False) -> T:
    r"""Return the minimum of the valid steps of every tensor along the sequence dimension.

    ``mask`` is one tensor shared by all the leaves. See
    ``batchtensor.tensor.masked_amin_along_seq`` for the semantics.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        data: The input data. Each leaf must be a ``torch.Tensor`` of shape
            ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is shared by all the leaves.
        keepdim: Whether the output tensors have dim retained or not.

    Returns:
        A structure with the same layout as ``data``. Each leaf is the minimum of the valid steps.

    Raises:
        TypeError: if a leaf is not a ``torch.Tensor``.
        ValueError: if ``mask`` is not a boolean tensor, if it is not 2-D,
            or if its shape is not the two first dimensions of a leaf.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import masked_amin_along_seq
        >>> from batchtensor.tensor import lengths_to_mask
        >>> data = {"a": torch.tensor([[1, 2, 3], [4, 5, 6]]), "b": [torch.tensor([[1, 2, 3], [4, 5, 6]])]}
        >>> mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
        >>> out = masked_amin_along_seq(data, mask)
        >>> out
        {'a': tensor([1, 4]), 'b': [tensor([1, 4])]}

        ```

    See Also:
        ``batchtensor.tensor.masked_amin_along_seq``: The same function for tensors.
    """
    return apply_to_tensors(data, partial(bt.masked_amin_along_seq, mask=mask, keepdim=keepdim))
