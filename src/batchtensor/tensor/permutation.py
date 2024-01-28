r"""Contain some functions to permute data in tensors."""

from __future__ import annotations

__all__ = ["permute_along_batch", "permute_along_seq"]

from typing import TYPE_CHECKING

from batchtensor.constants import BATCH_DIM, SEQ_DIM

if TYPE_CHECKING:
    import torch


def permute_along_batch(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    r"""Permute the tensor along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensor: The tensor to split.
        permutation: The 1-D tensor containing the indices of the
            permutation. The shape should match the batch dimension
            of the tensor.

    Returns:
        The tensor with permuted data along the batch dimension.

    Raises:
        RuntimeError: if the shape of the permutation does not match
            the batch dimension of the tensor.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import permute_along_batch
    >>> tensor = torch.arange(10).view(5, 2)
    >>> out = permute_along_batch(tensor, torch.tensor([2, 1, 3, 0, 4]))
    >>> out
    tensor([[4, 5],
            [2, 3],
            [6, 7],
            [0, 1],
            [8, 9]])

    ```
    """
    if permutation.shape[0] != tensor.shape[0]:
        msg = (
            f"permutation shape ({permutation.shape}) is not compatible with tensor shape "
            f"({tensor.shape})"
        )
        raise RuntimeError(msg)
    return tensor.index_select(dim=BATCH_DIM, index=permutation)


def permute_along_seq(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    r"""Permute the tensor along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The tensor to split.
        permutation: The 1-D tensor containing the indices of the
            permutation. The shape should match the sequence dimension
            of the tensor.

    Returns:
        The tensor with permuted data along the sequence dimension.

    Raises:
        RuntimeError: if the shape of the permutation does not match
            the sequence dimension of the tensor.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import permute_along_seq
    >>> tensor = torch.arange(10).view(2, 5)
    >>> out = permute_along_seq(tensor, torch.tensor([2, 1, 3, 0, 4]))
    >>> out
    tensor([[2, 1, 3, 0, 4],
            [7, 6, 8, 5, 9]])

    ```
    """
    if permutation.shape[0] != tensor.shape[1]:
        msg = (
            f"permutation shape ({permutation.shape}) is not compatible with tensor shape "
            f"({tensor.shape})"
        )
        raise RuntimeError(msg)
    return tensor.index_select(dim=SEQ_DIM, index=permutation)
