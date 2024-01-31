r"""Contain some reduction functions for tensors."""

from __future__ import annotations

__all__ = ["sum_along_batch", "sum_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def sum_along_batch(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the sum of all elements along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The sum of all elements along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import sum_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = sum_along_batch(tensor)
    >>> out
    tensor([20, 25])
    >>> out = sum_along_batch(tensor, keepdim=True)
    >>> out
    tensor([[20, 25]])

    ```
    """
    return torch.sum(tensor, dim=BATCH_DIM, keepdim=keepdim)


def sum_along_seq(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the sum of all elements along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The sum of all elements along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import sum_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = sum_along_seq(tensor)
    >>> out
    tensor([10, 35])
    >>> out = sum_along_seq(tensor, keepdim=True)
    >>> out
    tensor([[10], [35]])

    ```
    """
    return torch.sum(tensor, dim=SEQ_DIM, keepdim=keepdim)
