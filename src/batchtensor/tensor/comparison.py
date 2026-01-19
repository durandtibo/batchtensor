r"""Implements comparison functions for tensors."""

from __future__ import annotations

__all__ = ["argsort_along_batch", "argsort_along_seq", "sort_along_batch", "sort_along_seq"]

from typing import Any

import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def argsort_along_batch(
    tensor: torch.Tensor, descending: bool = False, **kwargs: Any
) -> torch.Tensor:
    r"""Return the indices that sort a tensor along the batch dimension
    in ascending order by value.

    This function returns the indices needed to sort the tensor along the
    batch dimension. The actual sorted values can be obtained by using
    ``sort_along_batch`` or by indexing with the returned indices.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The input tensor to sort.
        descending: Controls the sorting order. If ``False`` (default),
            sorts in ascending order. If ``True``, sorts in descending
            order.
        kwargs: Additional keyword arguments passed to ``torch.argsort``.
            Supports arguments like ``stable`` for stable sorting.

    Returns:
        A tensor of indices that would sort the input tensor along the
            batch dimension. Has the same shape as the input tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import argsort_along_batch
        >>> tensor = torch.tensor([[2, 6], [0, 3], [4, 9], [8, 1], [5, 7]])
        >>> # Sort in ascending order
        >>> out = argsort_along_batch(tensor)
        >>> out
        tensor([[1, 3], [0, 1], [2, 0], [4, 4], [3, 2]])
        >>> # Sort in descending order
        >>> out = argsort_along_batch(tensor, descending=True)
        >>> out
        tensor([[3, 2], [4, 4], [2, 0], [0, 1], [1, 3]])

        ```

    See Also:
        ``sort_along_batch``: Returns both sorted values and indices.
        ``argsort_along_seq``: Sort indices along sequence dimension.
        ``torch.argsort``: PyTorch's general argsort function.
    """
    return torch.argsort(tensor, dim=BATCH_DIM, descending=descending, **kwargs)


def argsort_along_seq(
    tensor: torch.Tensor, descending: bool = False, **kwargs: Any
) -> torch.Tensor:
    r"""Return the indices that sort a tensor along the sequence
    dimension in ascending order by value.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        descending: Controls the sorting order (ascending or
            descending).
        kwargs: Additional keywords arguments for ``torch.argsort``.

    Returns:
        The indices that sort a tensor along the sequence dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import argsort_along_seq
        >>> tensor = torch.tensor([[7, 3, 0, 8, 5], [1, 9, 6, 4, 2]])
        >>> out = argsort_along_seq(tensor)
        >>> out
        tensor([[2, 1, 4, 0, 3],
                [0, 4, 3, 2, 1]])
        >>> out = argsort_along_seq(tensor, descending=True)
        >>> out
        tensor([[3, 0, 4, 1, 2],
                [1, 2, 3, 4, 0]])

        ```
    """
    return torch.argsort(tensor, dim=SEQ_DIM, descending=descending, **kwargs)


def sort_along_batch(
    tensor: torch.Tensor, descending: bool = False, **kwargs: Any
) -> torch.return_types.sort:
    r"""Sort the elements of the input tensor along the batch dimension
    in ascending order by value.

    This function sorts the tensor along the batch dimension and returns
    both the sorted values and the corresponding indices. This is useful
    when you need both the sorted data and the permutation that created it.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The input tensor to sort.
        descending: Controls the sorting order. If ``False`` (default),
            sorts in ascending order. If ``True``, sorts in descending
            order.
        kwargs: Additional keyword arguments passed to ``torch.sort``.
            Supports arguments like ``stable`` for stable sorting.

    Returns:
        A namedtuple of (values, indices), where:
            - ``values``: The sorted tensor along the batch dimension.
            - ``indices``: The indices that would produce the sorted values
              from the original tensor.
        Both tensors have the same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import sort_along_batch
        >>> tensor = torch.tensor([[2, 6], [0, 3], [4, 9], [8, 1], [5, 7]])
        >>> out = sort_along_batch(tensor)
        >>> out
        torch.return_types.sort(
        values=tensor([[0, 1], [2, 3], [4, 6], [5, 7], [8, 9]]),
        indices=tensor([[1, 3], [0, 1], [2, 0], [4, 4], [3, 2]]))
        >>> # Access sorted values and indices
        >>> out.values
        tensor([[0, 1], [2, 3], [4, 6], [5, 7], [8, 9]])
        >>> out.indices
        tensor([[1, 3], [0, 1], [2, 0], [4, 4], [3, 2]])
        >>> # Sort in descending order
        >>> out = sort_along_batch(tensor, descending=True)
        >>> out
        torch.return_types.sort(
        values=tensor([[8, 9], [5, 7], [4, 6], [2, 3], [0, 1]]),
        indices=tensor([[3, 2], [4, 4], [2, 0], [0, 1], [1, 3]]))

        ```

    See Also:
        ``argsort_along_batch``: Returns only the sorting indices.
        ``sort_along_seq``: Sort along the sequence dimension.
        ``torch.sort``: PyTorch's general sort function.
    """
    return torch.sort(tensor, dim=BATCH_DIM, descending=descending, **kwargs)


def sort_along_seq(
    tensor: torch.Tensor, descending: bool = False, **kwargs: Any
) -> torch.return_types.sort:
    r"""Sort the elements of the input tensor along the sequence
    dimension in ascending order by value.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The input tensor.
        descending: Controls the sorting order (ascending or
            descending).
        kwargs: Additional keywords arguments for ``torch.sort``.

    Returns:
        A namedtuple of (values, indices), where the values are the
            sorted values and indices are the indices of the elements
            in the original input tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import sort_along_seq
        >>> tensor = torch.tensor([[7, 3, 0, 8, 5], [1, 9, 6, 4, 2]])
        >>> out = sort_along_seq(tensor)
        >>> out
        torch.return_types.sort(
        values=tensor([[0, 3, 5, 7, 8], [1, 2, 4, 6, 9]]),
        indices=tensor([[2, 1, 4, 0, 3], [0, 4, 3, 2, 1]]))
        >>> out = sort_along_seq(tensor, descending=True)
        >>> out
        torch.return_types.sort(
        values=tensor([[8, 7, 5, 3, 0], [9, 6, 4, 2, 1]]),
        indices=tensor([[3, 0, 4, 1, 2], [1, 2, 3, 4, 0]]))

        ```
    """
    return torch.sort(tensor, dim=SEQ_DIM, descending=descending, **kwargs)
