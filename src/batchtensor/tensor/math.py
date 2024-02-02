r"""Contain some mathematical functions for tensors."""

from __future__ import annotations

__all__ = ["cumsum_along_batch", "cumsum_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def cumsum_along_batch(tensor: torch.Tensor) -> torch.Tensor:
    r"""Return the cumulative sum of elements of input in the batch
    dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.

    Returns:
        The cumulative sum of elements of input in the batch
            dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import cumsum_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = cumsum_along_batch(tensor)
    >>> out
    tensor([[ 0,  1], [ 2,  4], [ 6,  9], [12, 16], [20, 25]])

    ```
    """
    return tensor.cumsum(dim=BATCH_DIM)


def cumsum_along_seq(tensor: torch.Tensor) -> torch.Tensor:
    r"""Return the cumulative sum of elements of input in the sequence
    dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.

    Returns:
        The cumulative sum of elements of input in the sequence
            dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import cumsum_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = cumsum_along_seq(tensor)
    >>> out
    tensor([[ 0,  1,  3,  6, 10],
            [ 5, 11, 18, 26, 35]])

    ```
    """
    return torch.cumsum(tensor, dim=SEQ_DIM)
