r"""Contain some tensor point-wise functions for nested data."""

from __future__ import annotations

__all__ = ["log", "log2", "log10", "log1p"]

from typing import Any

import torch

from batchtensor.recursive import recursive_apply


def log(data: Any) -> Any:
    r"""Return new tensors with the natural logarithm of the elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The natural logarithm of the elements. The output has the same
            structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import log
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = log(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.log)


def log2(data: Any) -> Any:
    r"""Return new tensors with the logarithm to the base 2 of the
    elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The logarithm to the base 2 of the elements. The output has
            the same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import log2
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = log2(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.log2)


def log10(data: Any) -> Any:
    r"""Return new tensors with the logarithm to the base 10 of the
    elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The with the logarithm to the base 10 of the elements. The
            output has the same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import log10
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = log10(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.log10)


def log1p(data: Any) -> Any:
    r"""Return new tensors with the natural logarithm of ``(1 + input)``.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The natural logarithm of ``(1 + input)``. The output has the
            same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import log1p
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = log1p(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.log1p)
