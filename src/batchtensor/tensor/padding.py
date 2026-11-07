r"""Implements padding functions for tensors."""

from __future__ import annotations

__all__ = ["pad_along_batch", "pad_along_seq"]

import math

import torch.nn.functional

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def pad_along_batch(tensor: torch.Tensor, pad_size: int, *, value: float = 0.0) -> torch.Tensor:
    r"""Pad the tensor along the batch dimension.

    The padding is added at the end of the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The tensor to pad. Must have at least one dimension.
        pad_size: The number of items to add to the batch dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items. It must be
            representable in the dtype of the tensor.

    Returns:
        The padded tensor. If the input has batch size ``b``, the output
            has batch size ``b + pad_size``.

    Raises:
        ValueError: if ``pad_size`` is negative or if ``value`` cannot
            be represented in the dtype of the tensor.

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
    return _pad(tensor, dim=BATCH_DIM, pad_size=pad_size, value=value)


def pad_along_seq(tensor: torch.Tensor, pad_size: int, *, value: float = 0.0) -> torch.Tensor:
    r"""Pad the tensor along the sequence dimension.

    The padding is added at the end of the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension (index 1).

    Args:
        tensor: The tensor to pad. Must have at least two dimensions.
        pad_size: The number of items to add to the sequence dimension.
            Must be a non-negative integer.
        value: The fill value used for the padded items. It must be
            representable in the dtype of the tensor.

    Returns:
        The padded tensor. If the input has sequence length ``s``, the
            output has sequence length ``s + pad_size``.

    Raises:
        ValueError: if ``pad_size`` is negative or if ``value`` cannot
            be represented in the dtype of the tensor.

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
    return _pad(tensor, dim=SEQ_DIM, pad_size=pad_size, value=value)


def _pad(tensor: torch.Tensor, dim: int, pad_size: int, value: float) -> torch.Tensor:
    r"""Pad a tensor at the end of a dimension.

    Args:
        tensor: The tensor to pad.
        dim: The dimension to pad.
        pad_size: The number of items to add. Must be non-negative.
        value: The fill value. It must be representable in the dtype
            of the tensor.

    Returns:
        The padded tensor.

    Raises:
        ValueError: if ``pad_size`` is negative or if ``value`` cannot
            be represented in the dtype of the tensor.
    """
    if pad_size < 0:
        msg = f"pad_size must be a non-negative integer but received {pad_size}"
        raise ValueError(msg)
    _check_fill_value(value, tensor.dtype)
    if pad_size == 0:
        return tensor
    # ``F.pad`` expects the padding widths starting from the last dimension,
    # so the target dimension index is counted from the end.
    ndim = tensor.dim()
    pad = [0, 0] * (ndim - dim - 1) + [0, pad_size]
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
