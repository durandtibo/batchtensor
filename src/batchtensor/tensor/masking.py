r"""Implements masking functions related to batches of padded
sequences."""

from __future__ import annotations

__all__ = [
    "lengths_to_mask",
    "masked_amax_along_seq",
    "masked_amin_along_seq",
    "masked_mean_along_seq",
    "masked_sum_along_seq",
]

import torch

from batchtensor.constants import SEQ_DIM


def lengths_to_mask(lengths: torch.Tensor, *, max_len: int | None = None) -> torch.Tensor:
    r"""Convert a tensor of sequence lengths to a boolean mask.

    This is typically used together with ``pad_along_seq`` to know which
    positions of a padded batch of sequences are valid (``True``) and
    which ones are padding (``False``).

    Args:
        lengths: A 1-d tensor of shape ``(batch_size,)`` containing the
            length of each sequence in the batch. It must have an
            integer dtype. A negative length gives a row of ``False``.
        max_len: The length of the sequence (i.e. second) dimension of
            the output mask. If ``None``, the maximum value in
            ``lengths`` is used. If it is smaller than a length, the
            row is truncated to ``max_len``.

    Returns:
        A boolean tensor of shape ``(batch_size, max_len)`` where
            ``mask[i, j]`` is ``True`` if ``j < lengths[i]`` and
            ``False`` otherwise.

    Raises:
        ValueError: if ``lengths`` is not a 1-D tensor or does not have
            an integer dtype.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import lengths_to_mask
        >>> out = lengths_to_mask(torch.tensor([3, 1, 2]))
        >>> out
        tensor([[ True,  True,  True],
                [ True, False, False],
                [ True,  True, False]])
        >>> out = lengths_to_mask(torch.tensor([3, 1, 2]), max_len=4)
        >>> out
        tensor([[ True,  True,  True, False],
                [ True, False, False, False],
                [ True,  True, False, False]])

        ```

    See Also:
        ``pad_along_seq``: Pad tensors along the sequence dimension.
    """
    if lengths.ndim != 1:
        msg = f"lengths must be a 1-D tensor but received a tensor with {lengths.ndim} dimensions"
        raise ValueError(msg)
    if lengths.is_floating_point() or lengths.is_complex() or lengths.dtype == torch.bool:
        msg = f"lengths must be an integer tensor but received dtype {lengths.dtype}"
        raise ValueError(msg)
    if max_len is None:
        max_len = max(int(lengths.max().item()), 0) if lengths.numel() > 0 else 0
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def masked_sum_along_seq(
    tensor: torch.Tensor, mask: torch.Tensor, *, keepdim: bool = False
) -> torch.Tensor:
    r"""Return the sum of the valid steps along the sequence dimension.

    The padded steps (``mask`` is ``False``) are replaced by ``0`` before
    the sum, i.e. this is ``tensor.masked_fill(~mask, 0).sum(dim=1)``.

    Note:
        This function assumes the sequence dimension is the second
            dimension. It is a fixed composition of torch operations,
            not a single torch call.

    Args:
        tensor: The input tensor of shape ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is broadcast over the trailing dimensions of
            ``tensor``.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The sum of the valid steps. A row without valid step gives ``0``.

    Raises:
        ValueError: if ``mask`` is not a boolean tensor, if it is not
            2-D, or if its shape is not the two first dimensions of
            ``tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import lengths_to_mask, masked_sum_along_seq
        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> mask = lengths_to_mask(torch.tensor([3, 1]))
        >>> out = masked_sum_along_seq(tensor, mask)
        >>> out
        tensor([6, 4])
        >>> out = masked_sum_along_seq(tensor, mask, keepdim=True)
        >>> out
        tensor([[6], [4]])

        ```

    See Also:
        ``lengths_to_mask``: Build the mask of a padded batch.
        ``sum_along_seq``: Sum all the steps.
    """
    return _fill_invalid(tensor, mask, 0).sum(dim=SEQ_DIM, keepdim=keepdim)


def masked_mean_along_seq(
    tensor: torch.Tensor, mask: torch.Tensor, *, keepdim: bool = False
) -> torch.Tensor:
    r"""Return the mean of the valid steps along the sequence dimension.

    This is the sum of the valid steps divided by the number of valid
    steps.

    Note:
        This function assumes the sequence dimension is the second
            dimension. It is a fixed composition of torch operations,
            not a single torch call.

    Args:
        tensor: The input tensor of shape ``(batch_size, seq_len, ...)``.
            It must have a floating point or complex dtype.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is broadcast over the trailing dimensions of
            ``tensor``.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The mean of the valid steps. A row without valid step gives
            ``nan`` (0 / 0, like the mean of an empty tensor).

    Raises:
        ValueError: if ``tensor`` does not have a floating point or
            complex dtype, if ``mask`` is not a boolean tensor, if it is
            not 2-D, or if its shape is not the two first dimensions of
            ``tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import lengths_to_mask, masked_mean_along_seq
        >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> mask = lengths_to_mask(torch.tensor([3, 1]))
        >>> out = masked_mean_along_seq(tensor, mask)
        >>> out
        tensor([2., 4.])
        >>> out = masked_mean_along_seq(tensor, mask, keepdim=True)
        >>> out
        tensor([[2.], [4.]])

        ```

    See Also:
        ``lengths_to_mask``: Build the mask of a padded batch.
        ``mean_along_seq``: Average all the steps.
    """
    if not (tensor.is_floating_point() or tensor.is_complex()):
        msg = (
            "masked_mean_along_seq requires a floating point or complex tensor "
            f"but received dtype {tensor.dtype}"
        )
        raise ValueError(msg)
    total = _fill_invalid(tensor, mask, 0).sum(dim=SEQ_DIM, keepdim=keepdim)
    count = mask.sum(dim=SEQ_DIM).reshape(mask.shape[0], *([1] * (total.ndim - 1)))
    return total / count.to(total.dtype)


def masked_amax_along_seq(
    tensor: torch.Tensor, mask: torch.Tensor, *, keepdim: bool = False
) -> torch.Tensor:
    r"""Return the maximum of the valid steps along the sequence dimension.

    The padded steps (``mask`` is ``False``) are replaced by the lowest
    value of the dtype of ``tensor`` before the reduction.

    Note:
        This function assumes the sequence dimension is the second
            dimension. It is a fixed composition of torch operations,
            not a single torch call.

    Args:
        tensor: The input tensor of shape ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is broadcast over the trailing dimensions of
            ``tensor``.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The maximum of the valid steps. A row without valid step gives
            the lowest value of the dtype (``False`` for a boolean
            tensor).

    Raises:
        ValueError: if ``mask`` is not a boolean tensor, if it is not
            2-D, or if its shape is not the two first dimensions of
            ``tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import lengths_to_mask, masked_amax_along_seq
        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
        >>> out = masked_amax_along_seq(tensor, mask)
        >>> out
        tensor([2, 4])

        ```

    See Also:
        ``lengths_to_mask``: Build the mask of a padded batch.
        ``amax_along_seq``: Maximum of all the steps.
    """
    return _fill_invalid(tensor, mask, _lowest(tensor.dtype)).amax(dim=SEQ_DIM, keepdim=keepdim)


def masked_amin_along_seq(
    tensor: torch.Tensor, mask: torch.Tensor, *, keepdim: bool = False
) -> torch.Tensor:
    r"""Return the minimum of the valid steps along the sequence dimension.

    The padded steps (``mask`` is ``False``) are replaced by the highest
    value of the dtype of ``tensor`` before the reduction.

    Note:
        This function assumes the sequence dimension is the second
            dimension. It is a fixed composition of torch operations,
            not a single torch call.

    Args:
        tensor: The input tensor of shape ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``, e.g.
            the output of ``lengths_to_mask``. ``True`` marks a valid
            step. It is broadcast over the trailing dimensions of
            ``tensor``.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The minimum of the valid steps. A row without valid step gives
            the highest value of the dtype (``True`` for a boolean
            tensor).

    Raises:
        ValueError: if ``mask`` is not a boolean tensor, if it is not
            2-D, or if its shape is not the two first dimensions of
            ``tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import lengths_to_mask, masked_amin_along_seq
        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> mask = lengths_to_mask(torch.tensor([2, 1]), max_len=3)
        >>> out = masked_amin_along_seq(tensor, mask)
        >>> out
        tensor([1, 4])

        ```

    See Also:
        ``lengths_to_mask``: Build the mask of a padded batch.
        ``amin_along_seq``: Minimum of all the steps.
    """
    return _fill_invalid(tensor, mask, _highest(tensor.dtype)).amin(dim=SEQ_DIM, keepdim=keepdim)


def _fill_invalid(tensor: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    r"""Replace the padded steps of a tensor by a value.

    Args:
        tensor: The input tensor of shape ``(batch_size, seq_len, ...)``.
        mask: A boolean tensor of shape ``(batch_size, seq_len)``.
            ``True`` marks a valid step.
        value: The value that replaces the steps where ``mask`` is
            ``False``.

    Returns:
        A new tensor with the same shape as ``tensor``.

    Raises:
        ValueError: if ``mask`` is not a boolean tensor, if it is not
            2-D, or if its shape is not the two first dimensions of
            ``tensor``.
    """
    if mask.dtype != torch.bool:
        msg = f"mask must be a boolean tensor but received dtype {mask.dtype}"
        raise ValueError(msg)
    if mask.ndim != 2:
        msg = f"mask must be a 2-D tensor but received a tensor with {mask.ndim} dimensions"
        raise ValueError(msg)
    if tensor.shape[:2] != mask.shape:
        msg = (
            f"mask must have the shape (batch_size, seq_len) of the tensor {tuple(tensor.shape[:2])} "
            f"but received {tuple(mask.shape)}"
        )
        raise ValueError(msg)
    valid = mask.reshape(*mask.shape, *([1] * (tensor.ndim - 2)))
    return tensor.masked_fill(~valid, value)


def _lowest(dtype: torch.dtype) -> float:
    r"""Return the lowest value of a dtype.

    Args:
        dtype: The dtype.

    Returns:
        The lowest finite value of ``dtype`` (``False`` for a boolean dtype).
    """
    if dtype == torch.bool:
        return False
    if dtype.is_floating_point:
        return torch.finfo(dtype).min
    return torch.iinfo(dtype).min


def _highest(dtype: torch.dtype) -> float:
    r"""Return the highest value of a dtype.

    Args:
        dtype: The dtype.

    Returns:
        The highest finite value of ``dtype`` (``True`` for a boolean dtype).
    """
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return torch.finfo(dtype).max
    return torch.iinfo(dtype).max
