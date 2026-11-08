r"""Implements padding functions for tensors."""

from __future__ import annotations

__all__ = ["pad_along_batch", "pad_along_seq", "pad_sequences"]

import math
from typing import TYPE_CHECKING, Literal

import torch.nn.functional

from batchtensor.constants import BATCH_DIM, SEQ_DIM

if TYPE_CHECKING:
    from collections.abc import Sequence

Side = Literal["left", "right"]


def pad_along_batch(
    tensor: torch.Tensor, pad_size: int, *, value: float = 0.0, side: Side = "right"
) -> torch.Tensor:
    r"""Pad the tensor along the batch dimension.

    The padding is added at the end of the batch dimension by default,
    or at the start with ``side="left"``.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The tensor to pad. Must have at least one dimension.
        pad_size: The number of items to add to the batch dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items. It must be
            representable in the dtype of the tensor.
        side: The side of the dimension where the padding is added,
            ``"right"`` (end) or ``"left"`` (start).

    Returns:
        The padded tensor. If the input has batch size ``b``, the output
            has batch size ``b + pad_size``.

    Raises:
        ValueError: if ``pad_size`` is negative, if ``value`` cannot
            be represented in the dtype of the tensor, or if ``side``
            is not ``"left"`` or ``"right"``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import pad_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5]])
        >>> out = pad_along_batch(tensor, pad_size=2)
        >>> out
        tensor([[0, 1],
                [2, 3],
                [4, 5],
                [0, 0],
                [0, 0]])

        ```

    See Also:
        ``pad_along_seq``: Pad along the sequence dimension instead.
        ``cat_along_batch``: Concatenate tensors along the batch dimension.
    """
    return _pad(tensor, dim=BATCH_DIM, pad_size=pad_size, value=value, side=side)


def pad_along_seq(
    tensor: torch.Tensor, pad_size: int, *, value: float = 0.0, side: Side = "right"
) -> torch.Tensor:
    r"""Pad the tensor along the sequence dimension.

    The padding is added at the end of the sequence dimension by default,
    or at the start with ``side="left"``.

    Note:
        This function assumes the sequence dimension is the second
            dimension (index 1).

    Args:
        tensor: The tensor to pad. Must have at least two dimensions.
        pad_size: The number of items to add to the sequence dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items. It must be
            representable in the dtype of the tensor.
        side: The side of the dimension where the padding is added,
            ``"right"`` (end) or ``"left"`` (start).

    Returns:
        The padded tensor. If the input has sequence length ``s``, the
            output has sequence length ``s + pad_size``.

    Raises:
        ValueError: if ``pad_size`` is negative, if ``value`` cannot
            be represented in the dtype of the tensor, or if ``side``
            is not ``"left"`` or ``"right"``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import pad_along_seq
        >>> tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])
        >>> out = pad_along_seq(tensor, pad_size=2)
        >>> out
        tensor([[0, 1, 2, 0, 0],
                [3, 4, 5, 0, 0]])

        ```

    See Also:
        ``pad_along_batch``: Pad along the batch dimension instead.
        ``cat_along_seq``: Concatenate tensors along the sequence dimension.
        ``lengths_to_mask``: Build a mask that flags the padded positions.
    """
    return _pad(tensor, dim=SEQ_DIM, pad_size=pad_size, value=value, side=side)


def _pad(tensor: torch.Tensor, dim: int, pad_size: int, value: float, side: Side) -> torch.Tensor:
    r"""Pad a tensor at the end of a dimension.

    Args:
        tensor: The tensor to pad.
        dim: The dimension to pad.
        pad_size: The number of items to add. Must be non-negative.
        value: The fill value. It must be representable in the dtype
            of the tensor.
        side: The side of the dimension where the padding is added.

    Returns:
        The padded tensor.

    Raises:
        ValueError: if ``pad_size`` is negative, if ``value`` cannot
            be represented in the dtype of the tensor, or if ``side`` is
            not ``"left"`` or ``"right"``.
    """
    if side not in ("left", "right"):
        msg = f"side must be 'left' or 'right' but received {side!r}"
        raise ValueError(msg)
    if pad_size < 0:
        msg = f"pad_size must be a non-negative integer but received {pad_size}"
        raise ValueError(msg)
    _check_fill_value(value, tensor.dtype)
    if pad_size == 0:
        return tensor
    # ``F.pad`` expects the padding widths starting from the last dimension,
    # so the target dimension index is counted from the end.
    ndim = tensor.dim()
    widths = [pad_size, 0] if side == "left" else [0, pad_size]
    pad = [0, 0] * (ndim - dim - 1) + widths
    return torch.nn.functional.pad(tensor, pad, value=value)


def _check_fill_value(value: float, dtype: torch.dtype) -> None:
    r"""Check that a fill value can be represented in a dtype.

    Args:
        value: The fill value.
        dtype: The dtype of the tensor to fill.

    Raises:
        ValueError: if ``dtype`` is a boolean dtype and ``value`` is
            not 0 or 1, or if ``dtype`` is an integer dtype and
            ``value`` is not an integer in the range of the dtype.
    """
    if dtype == torch.bool:
        if value not in (0, 1):
            msg = f"value ({value}) cannot be represented with dtype {dtype}"
            raise ValueError(msg)
        return
    if dtype.is_floating_point or dtype.is_complex:
        return
    info = torch.iinfo(dtype)
    if not math.isfinite(value) or int(value) != value:
        msg = f"value ({value}) cannot be represented with dtype {dtype}"
        raise ValueError(msg)
    if not info.min <= value <= info.max:
        msg = f"value ({value}) is out of range for dtype {dtype} [{info.min}, {info.max}]"
        raise ValueError(msg)


def pad_sequences(
    tensors: Sequence[torch.Tensor], *, value: float = 0.0, side: Side = "right"
) -> torch.Tensor:
    r"""Pad tensors of different lengths and stack them into a batch.

    Each tensor is a sequence of shape ``(seq_len_i, ...)``. The sequences
    can have different lengths along their first dimension but must have
    the same trailing shape. The output has shape
    ``(len(tensors), max_i(seq_len_i), ...)``. The lengths of the
    sequences are ``[t.shape[0] for t in tensors]``, and
    ``lengths_to_mask`` builds the mask of the padded batch.

    Note:
        The dtype and device of the output are the ones of the first tensor.
        The other tensors are copied into it, which casts them to this dtype
        (e.g. ``1.5`` becomes ``1`` when the first tensor is a ``long``
        tensor). No type promotion is done.

    Args:
        tensors: The sequences to pad. Each tensor must have at least one
            dimension and the tensors must have the same shape except for
            the first dimension.
        value: The fill value used for the padding. It must be
            representable in the dtype of the tensors.
        side: The side of the sequences where the padding is added,
            ``"right"`` (end) or ``"left"`` (start).

    Returns:
        The padded tensor of shape ``(batch_size, max_len, ...)``. The
            dtype and device are the ones of the first tensor.

    Raises:
        ValueError: if ``tensors`` is empty, if a tensor is a scalar, if the
            trailing shapes differ, if ``value`` cannot be represented in
            the dtype of the tensors, or if ``side`` is not ``"left"`` or
            ``"right"``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import pad_sequences
        >>> out = pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4])])
        >>> out
        tensor([[1, 2, 3],
                [4, 0, 0]])
        >>> out = pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4])], side="left")
        >>> out
        tensor([[1, 2, 3],
                [0, 0, 4]])

        ```

    See Also:
        ``lengths_to_mask``: Build the mask of a padded batch.
        ``pad_along_seq``: Pad a batch along the sequence dimension.
    """
    if side not in ("left", "right"):
        msg = f"side must be 'left' or 'right' but received {side!r}"
        raise ValueError(msg)
    if len(tensors) == 0:
        msg = "tensors must contain at least one tensor"
        raise ValueError(msg)
    first = tensors[0]
    if first.dim() == 0:
        msg = "tensors must have at least one dimension but received a scalar"
        raise ValueError(msg)
    for i, tensor in enumerate(tensors):
        if tensor.dim() == 0 or tensor.shape[1:] != first.shape[1:]:
            msg = (
                "tensors must have the same shape except for the first dimension but "
                f"tensor 0 has shape {tuple(first.shape)} and tensor {i} has shape "
                f"{tuple(tensor.shape)}"
            )
            raise ValueError(msg)
    _check_fill_value(value, first.dtype)
    max_len = max(tensor.shape[0] for tensor in tensors)
    out = first.new_full((len(tensors), max_len, *first.shape[1:]), value)
    for i, tensor in enumerate(tensors):
        length = tensor.shape[0]
        if side == "left":
            out[i, max_len - length :] = tensor
        else:
            out[i, :length] = tensor
    return out
