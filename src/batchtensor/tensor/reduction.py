r"""Contain some reduction functions for tensors."""

from __future__ import annotations

__all__ = [
    "amax_along_batch",
    "amax_along_seq",
    "amin_along_batch",
    "amin_along_seq",
    "argmax_along_batch",
    "argmax_along_seq",
    "argmin_along_batch",
    "argmin_along_seq",
    "sum_along_batch",
    "sum_along_seq",
]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def amax_along_batch(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the maximum of all elements along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The maximum of all elements along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import amax_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = amax_along_batch(tensor)
    >>> out
    tensor([8, 9])
    >>> out = amax_along_batch(tensor, keepdim=True)
    >>> out
    tensor([[8, 9]])

    ```
    """
    return torch.amax(tensor, dim=BATCH_DIM, keepdim=keepdim)


def amax_along_seq(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the maximum of all elements along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The maximum of all elements along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import amax_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = amax_along_seq(tensor)
    >>> out
    tensor([4, 9])
    >>> out = amax_along_seq(tensor, keepdim=True)
    >>> out
    tensor([[4], [9]])

    ```
    """
    return torch.amax(tensor, dim=SEQ_DIM, keepdim=keepdim)


def amin_along_batch(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the minimum of all elements along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The minimum of all elements along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import amin_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = amin_along_batch(tensor)
    >>> out
    tensor([0, 1])
    >>> out = amin_along_batch(tensor, keepdim=True)
    >>> out
    tensor([[0, 1]])

    ```
    """
    return torch.amin(tensor, dim=BATCH_DIM, keepdim=keepdim)


def amin_along_seq(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the minimum of all elements along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The minimum of all elements along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import amin_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = amin_along_seq(tensor)
    >>> out
    tensor([0, 5])
    >>> out = amin_along_seq(tensor, keepdim=True)
    >>> out
    tensor([[0], [5]])

    ```
    """
    return torch.amin(tensor, dim=SEQ_DIM, keepdim=keepdim)


def argmax_along_batch(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the indices of the maximum value of all elements along the
    batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the maximum value of all elements along the
            batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import argmax_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = argmax_along_batch(tensor)
    >>> out
    tensor([4, 4])
    >>> out = argmax_along_batch(tensor, keepdim=True)
    >>> out
    tensor([[4, 4]])

    ```
    """
    return torch.argmax(tensor, dim=BATCH_DIM, keepdim=keepdim)


def argmax_along_seq(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the indices of the maximum value of all elements along the
    sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the maximum value of all elements along the
            sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import argmax_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = argmax_along_seq(tensor)
    >>> out
    tensor([4, 4])
    >>> out = argmax_along_seq(tensor, keepdim=True)
    >>> out
    tensor([[4], [4]])

    ```
    """
    return torch.argmax(tensor, dim=SEQ_DIM, keepdim=keepdim)


def argmin_along_batch(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the indices of the minimum value of all elements along the
    batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the minimum value of all elements along the
            batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import argmin_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = argmin_along_batch(tensor)
    >>> out
    tensor([0, 0])
    >>> out = argmin_along_batch(tensor, keepdim=True)
    >>> out
    tensor([[0, 0]])

    ```
    """
    return torch.argmin(tensor, dim=BATCH_DIM, keepdim=keepdim)


def argmin_along_seq(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""Return the indices of the minimum value of all elements along the
    sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the minimum value of all elements along the
            sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import argmin_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = argmin_along_seq(tensor)
    >>> out
    tensor([0, 0])
    >>> out = argmin_along_seq(tensor, keepdim=True)
    >>> out
    tensor([[0], [0]])

    ```
    """
    return torch.argmin(tensor, dim=SEQ_DIM, keepdim=keepdim)


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
