r"""Contain some tensor slicing functions for nested data."""

from __future__ import annotations

__all__ = [
    "chunk_along_batch",
    "chunk_along_seq",
]


from typing import TYPE_CHECKING

from batchtensor import tensor as bt

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


def chunk_along_batch(
    data: dict[Hashable, torch.Tensor], chunks: int
) -> tuple[dict[Hashable, torch.Tensor], ...]:
    r"""Split all the tensors into chunks along the batch dimension.

    Each chunk is a view of the input tensor.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        chunks: Number of chunks to return.

    Returns:
        The data chuncks.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import chunk_along_batch
    >>> data = {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
    >>> outputs = chunk_along_batch(data, chunks=3)
    >>> outputs
    ({'a': tensor([[0, 1], [2, 3]]), 'b': tensor([4, 3])},
     {'a': tensor([[4, 5], [6, 7]]), 'b': tensor([2, 1])},
     {'a': tensor([[8, 9]]), 'b': tensor([0])})

    ```
    """
    keys = data.keys()
    return tuple(
        [
            dict(zip(keys, values))
            for values in zip(*[bt.chunk_along_batch(tensor, chunks) for tensor in data.values()])
        ]
    )


def chunk_along_seq(
    data: dict[Hashable, torch.Tensor], chunks: int
) -> tuple[dict[Hashable, torch.Tensor], ...]:
    r"""Split the tensor into chunks along the sequence dimension.

    Each chunk is a view of the input tensor.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        chunks: Number of chunks to return.

    Returns:
        The data chuncks.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import chunk_along_seq
    >>> data = {'a': torch.arange(10).view(2, 5), 'b': torch.tensor([[4, 3, 2, 1, 0]])}
    >>> outputs = chunk_along_seq(data, chunks=3)
    >>> outputs
    ({'a': tensor([[0, 1], [5, 6]]), 'b': tensor([[4, 3]])},
     {'a': tensor([[2, 3], [7, 8]]), 'b': tensor([[2, 1]])},
     {'a': tensor([[4], [9]]), 'b': tensor([[0]])})

    ```
    """
    keys = data.keys()
    return tuple(
        [
            dict(zip(keys, values))
            for values in zip(*[bt.chunk_along_seq(tensor, chunks) for tensor in data.values()])
        ]
    )
