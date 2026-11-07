r"""Implements indexing functions for tensors."""

from __future__ import annotations

__all__ = ["index_select_along_batch", "index_select_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def index_select_along_batch(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    r"""Return a new tensor which indexes the ``input`` tensor along the
    batch dimension using the entries in ``index`` which is a
    ``LongTensor``.

    This function selects specific batch items based on the provided indices.
    Unlike ``select_along_batch`` which selects a single item and reduces
    dimensionality, this function maintains the batch dimension and can select
    multiple items, duplicate items, or reorder items.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The input tensor. Must have at least one dimension.
        index: A 1-D tensor containing the indices to select. Must be a
            ``LongTensor``. Can contain duplicate indices to repeat batch
            items, or be shorter/longer than the batch dimension to select
            a subset or create a larger output.

    Returns:
        The indexed tensor along the batch dimension. The output has shape
            ``(index.size(0), *tensor.shape[1:])`` where the batch dimension
            size matches the length of the index tensor.

    Raises:
        ValueError: if ``index`` is not a 1-D tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import index_select_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> # Select specific batch items
        >>> out = index_select_along_batch(tensor, torch.tensor([2, 4]))
        >>> out
        tensor([[4, 5],
                [8, 9]])
        >>> # Reverse order
        >>> out = index_select_along_batch(tensor, torch.tensor([4, 3, 2, 1, 0]))
        >>> out
        tensor([[8, 9],
                [6, 7],
                [4, 5],
                [2, 3],
                [0, 1]])
        >>> # Duplicate batch items
        >>> out = index_select_along_batch(tensor, torch.tensor([0, 0, 1]))
        >>> out
        tensor([[0, 1],
                [0, 1],
                [2, 3]])

        ```

    See Also:
        ``select_along_batch``: Select a single batch item (reduces dimension).
        ``permute_along_batch``: Reorder all batch items with a permutation.
        ``slice_along_batch``: Select a contiguous range of batch items.
        ``index_select_along_seq``: Index select along sequence dimension.
    """
    if index.ndim != 1:
        msg = f"index must be a 1-D tensor but received a tensor with {index.ndim} dimensions"
        raise ValueError(msg)
    return tensor.index_select(dim=BATCH_DIM, index=index)


def index_select_along_seq(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    r"""Return a new tensor which indexes the ``input`` tensor along the
    sequence dimension using the entries in ``index`` which is a
    ``LongTensor``.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        index: The tensor containing the indices to index. A 1-D tensor
            of shape ``(n,)`` selects the same positions for every batch
            item. A 2-D tensor of shape ``(batch_size, n)`` selects
            different positions for each batch item.

    Returns:
        The indexed tensor along the sequence dimension.

    Raises:
        ValueError: if ``index`` is not a 1-D or 2-D tensor, or if its
            first dimension is not the batch size of ``tensor`` (2-D).

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import index_select_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = index_select_along_seq(tensor, torch.tensor([2, 4]))
        >>> out
        tensor([[2, 4],
                [7, 9]])
        >>> out = index_select_along_seq(tensor, torch.tensor([4, 3, 2, 1, 0]))
        >>> out
        tensor([[4, 3, 2, 1, 0],
                [9, 8, 7, 6, 5]])

        ```
    """
    if index.ndim == 1:
        return tensor.index_select(dim=SEQ_DIM, index=index)
    if index.ndim != 2:
        msg = (
            f"index must be a 1-D or 2-D tensor but received a tensor with {index.ndim} dimensions"
        )
        raise ValueError(msg)
    batch_size, seq_len = index.shape
    if batch_size != tensor.shape[BATCH_DIM]:
        msg = (
            f"index batch size ({batch_size}) is not compatible with "
            f"tensor batch size ({tensor.shape[BATCH_DIM]})"
        )
        raise ValueError(msg)
    batch_index = torch.arange(batch_size, device=index.device).repeat_interleave(seq_len)
    index = index.flatten().long()
    return tensor[batch_index, index].reshape(batch_size, seq_len, *tensor.shape[2:])
