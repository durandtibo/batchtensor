r"""Implements mathematical functions for tensors."""

from __future__ import annotations

__all__ = [
    "cumprod_along_batch",
    "cumprod_along_seq",
    "cumsum_along_batch",
    "cumsum_along_seq",
]

from typing import TYPE_CHECKING

from batchtensor.constants import BATCH_DIM, SEQ_DIM

if TYPE_CHECKING:
    import torch


def cumprod_along_batch(tensor: torch.Tensor) -> torch.Tensor:
    r"""Return the cumulative product of elements of input in the batch
    dimension.

    This function computes the cumulative product along the batch dimension,
    where each element in the output is the product of all elements up to
    that position in the batch. This is useful for computing running products
    or compound growth factors over batch items.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Warning:
        Cumulative products can quickly overflow for large tensors or
        grow very large. Consider using log-space computations if needed.

    Args:
        tensor: The input tensor. Must have at least one dimension.

    Returns:
        A tensor containing the cumulative product of elements along the
            batch dimension. Has the same shape and dtype as the input tensor.
            Element ``output[i]`` is the product of ``input[0]`` through
            ``input[i]`` along the batch dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import cumprod_along_batch
        >>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        >>> out = cumprod_along_batch(tensor)
        >>> out
        tensor([[   1,    2], [   3,    8], [  15,   48], [ 105,  384], [ 945, 3840]])
        >>> # Each row is the product of all previous rows
        >>> # Row 0: [1, 2]
        >>> # Row 1: [1, 2] * [3, 4] = [3, 8]
        >>> # Row 2: [3, 8] * [5, 6] = [15, 48]
        >>> # etc.

        ```

    See Also:
        ``cumsum_along_batch``: Cumulative sum instead of product.
        ``cumprod_along_seq``: Cumulative product along sequence dimension.
        ``prod_along_batch``: Total product (single value per feature).
        ``torch.cumprod``: PyTorch's general cumulative product function.
    """
    return tensor.cumprod(dim=BATCH_DIM)


def cumprod_along_seq(tensor: torch.Tensor) -> torch.Tensor:
    r"""Return the cumulative product of elements of input in the
    sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.

    Returns:
        The cumulative product of elements of input in the sequence
            dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import cumprod_along_seq
        >>> tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        >>> out = cumprod_along_seq(tensor)
        >>> out
        tensor([[    1,     2,     6,    24,   120],
                [    6,    42,   336,  3024, 30240]])

        ```
    """
    return tensor.cumprod(dim=SEQ_DIM)


def cumsum_along_batch(tensor: torch.Tensor) -> torch.Tensor:
    r"""Return the cumulative sum of elements of input in the batch
    dimension.

    This function computes the cumulative sum along the batch dimension,
    where each element in the output is the sum of all elements up to that
    position in the batch. This is useful for computing running totals or
    prefix sums over batch items.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The input tensor. Must have at least one dimension.

    Returns:
        A tensor containing the cumulative sum of elements along the batch
            dimension. Has the same shape and dtype as the input tensor.
            Element ``output[i]`` is the sum of ``input[0]`` through
            ``input[i]`` along the batch dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import cumsum_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = cumsum_along_batch(tensor)
        >>> out
        tensor([[ 0,  1], [ 2,  4], [ 6,  9], [12, 16], [20, 25]])
        >>> # Each row is the sum of all previous rows
        >>> # Row 0: [0, 1]
        >>> # Row 1: [0, 1] + [2, 3] = [2, 4]
        >>> # Row 2: [2, 4] + [4, 5] = [6, 9]
        >>> # etc.

        ```

    See Also:
        ``cumprod_along_batch``: Cumulative product instead of sum.
        ``cumsum_along_seq``: Cumulative sum along sequence dimension.
        ``sum_along_batch``: Total sum (single value per feature).
        ``torch.cumsum``: PyTorch's general cumulative sum function.
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

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import cumsum_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = cumsum_along_seq(tensor)
        >>> out
        tensor([[ 0,  1,  3,  6, 10],
                [ 5, 11, 18, 26, 35]])

        ```
    """
    return tensor.cumsum(dim=SEQ_DIM)
