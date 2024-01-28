r"""Contain some indexing functions for tensors."""

from __future__ import annotations

__all__ = ["index_select_along_batch", "index_select_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def index_select_along_batch(
    input: torch.Tensor, index: torch.Tensor  # noqa: A002
) -> torch.Tensor:
    r"""Return a new tensor which indexes the ``input`` tensor along the
    batch dimension using the entries in ``index`` which is a
    ``LongTensor``.

    Args:
        input: The input tensor.
        index: The 1-D tensor containing the indices to index.

    Returns:
        The indexed tensor along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import index_select_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> index_select_along_batch(tensor, torch.tensor([2, 4]))
    tensor([[4, 5],
            [8, 9]])
    >>> index_select_along_batch(tensor, torch.tensor([4, 3, 2, 1, 0]))
    tensor([[8, 9],
            [6, 7],
            [4, 5],
            [2, 3],
            [0, 1]])

    ```
    """
    return torch.index_select(input, BATCH_DIM, index)


def index_select_along_seq(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:  # noqa: A002
    r"""Return a new tensor which indexes the ``input`` tensor along the
    sequence dimension using the entries in ``index`` which is a
    ``LongTensor``.

    Args:
        input: The input tensor.
        index: The 1-D tensor containing the indices to index.

    Returns:
        The indexed tensor along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import index_select_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> index_select_along_seq(tensor, torch.tensor([2, 4]))
    tensor([[2, 4],
            [7, 9]])
    >>> index_select_along_seq(tensor, torch.tensor([4, 3, 2, 1, 0]))
    tensor([[4, 3, 2, 1, 0],
            [9, 8, 7, 6, 5]])

    ```
    """
    return torch.index_select(input, SEQ_DIM, index)
