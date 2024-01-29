r"""Contain some joining functions for tensors."""

from __future__ import annotations

__all__ = ["cat_along_batch", "cat_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def cat_along_batch(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    r"""Concatenate the given tensors in the batch dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the batch dimension is the first
            dimension.

    Args:
        tensors: Specifies the batches to concatenate.

    Returns:
        The concatenated tensors along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import cat_along_batch
    >>> tensors = [
    ...     torch.tensor([[0, 1, 2], [4, 5, 6]]),
    ...     torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ... ]
    >>> out = cat_along_batch(tensors)
    >>> out
    tensor([[ 0,  1,  2],
            [ 4,  5,  6],
            [10, 11, 12],
            [13, 14, 15]])

    ```
    """
    return torch.cat(tensors, dim=BATCH_DIM)


def cat_along_seq(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    r"""Concatenate the given tensors in the sequence dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensors: Specifies the batches to concatenate.

    Returns:
        The concatenated tensors along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import cat_along_seq
    >>> tensors = [
    ...     torch.tensor([[0, 1, 2], [4, 5, 6]]),
    ...     torch.tensor([[10, 11], [12, 13]]),
    ... ]
    >>> out = cat_along_seq(tensors)
    >>> out
    tensor([[ 0,  1,  2, 10, 11],
            [ 4,  5,  6, 12, 13]])

    ```
    """
    return torch.cat(tensors, dim=SEQ_DIM)
