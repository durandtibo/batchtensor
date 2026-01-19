r"""Implements slicing functions for tensors."""

from __future__ import annotations

__all__ = [
    "chunk_along_batch",
    "chunk_along_seq",
    "select_along_batch",
    "select_along_seq",
    "slice_along_batch",
    "slice_along_seq",
    "split_along_batch",
    "split_along_seq",
]


from typing import TYPE_CHECKING

from batchtensor.constants import BATCH_DIM, SEQ_DIM

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


def chunk_along_batch(tensor: torch.Tensor, chunks: int) -> tuple[torch.Tensor, ...]:
    r"""Split the tensor into chunks along the batch dimension.

    Each chunk is a view of the input tensor. This operation attempts to
    split the tensor into the specified number of chunks of approximately
    equal size. If the tensor size is not evenly divisible by the number
    of chunks, the last chunk will be smaller.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The tensor to split. Must have at least one dimension.
        chunks: Number of chunks to return. Must be a positive integer.
            If ``chunks`` is greater than the batch size, some chunks may
            be empty.

    Returns:
        A tuple of tensor chunks along the batch dimension. Each chunk is
            a view of the original tensor (shares memory).

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import chunk_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> outputs = chunk_along_batch(tensor, chunks=3)
        >>> outputs
        (tensor([[0, 1], [2, 3]]),
         tensor([[4, 5], [6, 7]]),
         tensor([[8, 9]]))

        ```

    See Also:
        ``split_along_batch``: Split with specified sizes for each chunk.
        ``slice_along_batch``: Extract a slice from the batch dimension.
        ``chunk_along_seq``: Split along the sequence dimension instead.
    """
    return tensor.chunk(chunks=chunks, dim=BATCH_DIM)


def chunk_along_seq(tensor: torch.Tensor, chunks: int) -> tuple[torch.Tensor, ...]:
    r"""Split the tensor into chunks along the sequence dimension.

    Each chunk is a view of the input tensor.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The tensor to split.
        chunks: Number of chunks to return.

    Returns:
        The tensor chunks.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import chunk_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> outputs = chunk_along_seq(tensor, chunks=3)
        >>> outputs
        (tensor([[0, 1], [5, 6]]),
         tensor([[2, 3], [7, 8]]),
         tensor([[4], [9]]))

        ```
    """
    return tensor.chunk(chunks=chunks, dim=SEQ_DIM)


def select_along_batch(tensor: torch.Tensor, index: int) -> torch.Tensor:
    r"""Slice the input tensor along the batch dimension at the given
    index.

    This function returns a view of the original tensor with the batch dimension
    removed. It selects a single element from the batch dimension, reducing the
    tensor's dimensionality by one.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The input tensor. Must have at least one dimension.
        index: The index to select. Can be negative for indexing from the
            end (e.g., -1 for the last batch item). Must be in the range
            ``[-batch_size, batch_size-1]``.

    Returns:
        The sliced tensor with the batch dimension removed. The shape is
            ``tensor.shape[1:]``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import select_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = select_along_batch(tensor, index=2)
        >>> out
        tensor([4, 5])
        >>> # Negative indexing from the end
        >>> out = select_along_batch(tensor, index=-1)
        >>> out
        tensor([8, 9])

        ```

    See Also:
        ``slice_along_batch``: Extract a range of batch items (keeps batch dimension).
        ``index_select_along_batch``: Select multiple indices (keeps batch dimension).
        ``select_along_seq``: Select from the sequence dimension instead.
    """
    return tensor[index]


def select_along_seq(tensor: torch.Tensor, index: int) -> torch.Tensor:
    r"""Slice the input tensor along the sequence dimension at the given
    index.

    This function returns a view of the original tensor with the sequence dimension removed.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        index: The index to select with.

    Returns:
        The sliced tensor along the sequence dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import select_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = select_along_seq(tensor, index=2)
        >>> out
        tensor([2, 7])

        ```
    """
    return tensor[:, index]


def slice_along_batch(
    tensor: torch.Tensor, start: int = 0, stop: int | None = None, step: int = 1
) -> torch.Tensor:
    r"""Slice the tensor along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        start: The index where the slicing of object starts.
        stop: The index where the slicing of object stops.
            ``None`` means last.
        step: The increment between each index for slicing.

    Returns:
        The sliced tensor along the batch dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import slice_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = slice_along_batch(tensor, start=2)
        >>> out
        tensor([[4, 5],
                [6, 7],
                [8, 9]])
        >>> out = slice_along_batch(tensor, stop=3)
        >>> out
        tensor([[0, 1],
                [2, 3],
                [4, 5]])
        >>> out = slice_along_batch(tensor, step=2)
        >>> out
        tensor([[0, 1],
                [4, 5],
                [8, 9]])

        ```
    """
    return tensor[start:stop:step]


def slice_along_seq(
    tensor: torch.Tensor, start: int = 0, stop: int | None = None, step: int = 1
) -> torch.Tensor:
    r"""Slice the tensor along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        start: The index where the slicing of object starts.
        stop: The index where the slicing of object stops.
            ``None`` means last.
        step: The increment between each index for slicing.

    Returns:
        The sliced tensor along the sequence dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import slice_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])
        >>> out = slice_along_seq(tensor, start=2)
        >>> out
        tensor([[2, 3, 4],
                [7, 6, 5]])
        >>> out = slice_along_seq(tensor, stop=3)
        >>> out
        tensor([[0, 1, 2],
                [9, 8, 7]])
        >>> out = slice_along_seq(tensor, step=2)
        >>> out
        tensor([[0, 2, 4],
                [9, 7, 5]])

        ```
    """
    return tensor[:, start:stop:step]


def split_along_batch(
    tensor: torch.Tensor, split_size_or_sections: int | Sequence[int]
) -> torch.Tensor:
    r"""Split the tensor into chunks along the batch dimension.

    Each chunk is a view of the original tensor.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        split_size_or_sections: Size of a single chunk or list of
            sizes for each chunk

    Returns:
        The tensor chunks.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import split_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> outputs = split_along_batch(tensor, split_size_or_sections=2)
        >>> outputs
        (tensor([[0, 1], [2, 3]]),
         tensor([[4, 5], [6, 7]]),
         tensor([[8, 9]]))

        ```
    """
    return tensor.split(split_size_or_sections, dim=BATCH_DIM)


def split_along_seq(
    tensor: torch.Tensor, split_size_or_sections: int | Sequence[int]
) -> torch.Tensor:
    r"""Split the tensor into chunks along the sequence dimension.

    Each chunk is a view of the original tensor.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        split_size_or_sections: Size of a single chunk or list of
            sizes for each chunk

    Returns:
        The tensor chunks.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import split_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> outputs = split_along_seq(tensor, split_size_or_sections=2)
        >>> outputs
        (tensor([[0, 1], [5, 6]]),
         tensor([[2, 3], [7, 8]]),
         tensor([[4], [9]]))

        ```
    """
    return tensor.split(split_size_or_sections, dim=SEQ_DIM)
