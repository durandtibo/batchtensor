r"""Contain some indexing functions for tensors."""

from __future__ import annotations

__all__ = [
    "chunk_along_batch",
    "chunk_along_seq",
    "select_along_batch",
    "select_along_seq",
    "slice_along_batch",
    "slice_along_seq",
]

from typing import TYPE_CHECKING

from batchtensor.constants import BATCH_DIM, SEQ_DIM

if TYPE_CHECKING:
    import torch


def chunk_along_batch(tensor: torch.Tensor, chunks: int) -> tuple[torch.Tensor, ...]:
    r"""Split the tensor into chunks along the batch dimension.

    Each chunk is a view of the input tensor.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The tensor to split.
        chunks: Number of chunks to return.

    Returns:
        The tensor chunks.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import chunk_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> chunk_along_batch(tensor, chunks=3)
    (tensor([[0, 1], [2, 3]]),
     tensor([[4, 5], [6, 7]]),
     tensor([[8, 9]]))

    ```
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

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import chunk_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> chunk_along_seq(tensor, chunks=3)
    (tensor([[0, 1], [5, 6]]),
     tensor([[2, 3], [7, 8]]),
     tensor([[4], [9]]))

     ```
    """
    return tensor.chunk(chunks=chunks, dim=SEQ_DIM)


def select_along_batch(tensor: torch.Tensor, index: int) -> torch.Tensor:
    r"""Slice the input tensor along the batch dimension at the given
    index.

    This function returns a view of the original tensor with the batch dimension removed.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        index: The index to select with.

    Returns:
        The sliced tensor along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import select_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> select_along_batch(tensor, index=2)
    tensor([4, 5])

    ```
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

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import select_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> select_along_seq(tensor, index=2)
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
        start: Specifies the index where the slicing of object starts.
        stop: Specifies the index where the slicing of object stops.
            ``None`` means last.
        step: Specifies the increment between each index for slicing.

    Returns:
        The sliced tensor along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import slice_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> slice_along_batch(tensor, start=2)
    tensor([[4, 5],
            [6, 7],
            [8, 9]])
    >>> slice_along_batch(tensor, stop=3)
    tensor([[0, 1],
            [2, 3],
            [4, 5]])
    >>> slice_along_batch(tensor, step=2)
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
        start: Specifies the index where the slicing of object starts.
        stop: Specifies the index where the slicing of object stops.
            ``None`` means last.
        step: Specifies the increment between each index for slicing.

    Returns:
        The sliced tensor along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import slice_along_seq
    >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])
    >>> slice_along_seq(tensor, start=2)
    tensor([[2, 3, 4],
            [7, 6, 5]])
    >>> slice_along_seq(tensor, stop=3)
    tensor([[0, 1, 2],
            [9, 8, 7]])
    >>> slice_along_seq(tensor, step=2)
    tensor([[0, 2, 4],
            [9, 7, 5]])

    ```
    """
    return tensor[:, start:stop:step]
