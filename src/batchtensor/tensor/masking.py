r"""Implements masking functions related to batches of padded
sequences."""

from __future__ import annotations

__all__ = ["lengths_to_mask"]

import torch


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
