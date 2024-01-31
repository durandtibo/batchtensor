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
]

from functools import partial
from typing import Any

from batchtensor import tensor as bt
from batchtensor.recursive import recursive_apply


def amax_along_batch(data: Any, keepdim: bool = False) -> Any:
    r"""Return the maximum of all elements along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The maximum of all elements along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import amax_along_batch
    >>> data = {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
    >>> out = amax_along_batch(data)
    >>> out
    {'a': tensor([8, 9]), 'b': tensor(4)}
    >>> out = amax_along_batch(data, keepdim=True)
    >>> out
    {'a': tensor([[8, 9]]), 'b': tensor([4])}

    ```
    """
    return recursive_apply(data, partial(bt.amax_along_batch, keepdim=keepdim))


def amax_along_seq(data: Any, keepdim: bool = False) -> Any:
    r"""Return the maximum of all elements along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The maximum of all elements along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import amax_along_seq
    >>> data = {'a': torch.arange(10).view(2, 5), 'b': torch.tensor([[4, 3, 2, 1, 0]])}
    >>> out = amax_along_seq(data)
    >>> out
    {'a': tensor([4, 9]), 'b': tensor([4])}
    >>> out = amax_along_seq(data, keepdim=True)
    >>> out
    {'a': tensor([[4], [9]]), 'b': tensor([[4]])}

    ```
    """
    return recursive_apply(data, partial(bt.amax_along_seq, keepdim=keepdim))


def amin_along_batch(data: Any, keepdim: bool = False) -> Any:
    r"""Return the minimum of all elements along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The minimum of all elements along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import amin_along_batch
    >>> data = {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
    >>> out = amin_along_batch(data)
    >>> out
    {'a': tensor([0, 1]), 'b': tensor(0)}
    >>> out = amin_along_batch(data, keepdim=True)
    >>> out
    {'a': tensor([[0, 1]]), 'b': tensor([0])}

    ```
    """
    return recursive_apply(data, partial(bt.amin_along_batch, keepdim=keepdim))


def amin_along_seq(data: Any, keepdim: bool = False) -> Any:
    r"""Return the minimum of all elements along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The minimum of all elements along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import amin_along_seq
    >>> data = {'a': torch.arange(10).view(2, 5), 'b': torch.tensor([[4, 3, 2, 1, 0]])}
    >>> out = amin_along_seq(data)
    >>> out
    {'a': tensor([0, 5]), 'b': tensor([0])}
    >>> out = amin_along_seq(data, keepdim=True)
    >>> out
    {'a': tensor([[0], [5]]), 'b': tensor([[0]])}

    ```
    """
    return recursive_apply(data, partial(bt.amin_along_seq, keepdim=keepdim))


def argmax_along_batch(data: Any, keepdim: bool = False) -> Any:
    r"""Return the indices of the maximum value of all elements along the
    batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the maximum value of all elements along the
            batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import argmax_along_batch
    >>> data = {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
    >>> out = argmax_along_batch(data)
    >>> out
    {'a': tensor([4, 4]), 'b': tensor(0)}
    >>> out = argmax_along_batch(data, keepdim=True)
    >>> out
    {'a': tensor([[4, 4]]), 'b': tensor([0])}

    ```
    """
    return recursive_apply(data, partial(bt.argmax_along_batch, keepdim=keepdim))


def argmax_along_seq(data: Any, keepdim: bool = False) -> Any:
    r"""Return the indices of the maximum value of all elements along the
    sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the maximum value of all elements along the
            sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import argmax_along_seq
    >>> data = {'a': torch.arange(10).view(2, 5), 'b': torch.tensor([[4, 3, 2, 1, 0]])}
    >>> out = argmax_along_seq(data)
    >>> out
    {'a': tensor([4, 4]), 'b': tensor([0])}
    >>> out = argmax_along_seq(data, keepdim=True)
    >>> out
    {'a': tensor([[4], [4]]), 'b': tensor([[0]])}

    ```
    """
    return recursive_apply(data, partial(bt.argmax_along_seq, keepdim=keepdim))


def argmin_along_batch(data: Any, keepdim: bool = False) -> Any:
    r"""Return the indices of the minimum value of all elements along the
    batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the minimum value of all elements along the
            batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import argmin_along_batch
    >>> data = {"a": torch.arange(10).view(5, 2), "b": torch.tensor([4, 3, 2, 1, 0])}
    >>> out = argmin_along_batch(data)
    >>> out
    {'a': tensor([0, 0]), 'b': tensor(4)}
    >>> out = argmin_along_batch(data, keepdim=True)
    >>> out
    {'a': tensor([[0, 0]]), 'b': tensor([4])}

    ```
    """
    return recursive_apply(data, partial(bt.argmin_along_batch, keepdim=keepdim))


def argmin_along_seq(data: Any, keepdim: bool = False) -> Any:
    r"""Return the indices of the minimum value of all elements along the
    sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The indices of the minimum value of all elements along the
            sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import argmin_along_seq
    >>> data = {'a': torch.arange(10).view(2, 5), 'b': torch.tensor([[4, 3, 2, 1, 0]])}
    >>> out = argmin_along_seq(data)
    >>> out
    {'a': tensor([0, 0]), 'b': tensor([4])}
    >>> out = argmin_along_seq(data, keepdim=True)
    >>> out
    {'a': tensor([[0], [0]]), 'b': tensor([[4]])}

    ```
    """
    return recursive_apply(data, partial(bt.argmin_along_seq, keepdim=keepdim))
