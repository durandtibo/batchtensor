r"""Contain some tensor point-wise functions for nested data."""

from __future__ import annotations

__all__ = ["log", "log2", "log10", "log1p", "exp", "exp2", "expm1"]

from typing import Any

import torch

from batchtensor.recursive import recursive_apply


def exp(data: Any) -> Any:
    r"""Return new tensors with the exponential of the elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The exponential of the elements. The output has the same
            structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import exp
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = exp(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.exp)


def exp2(data: Any) -> Any:
    r"""Return new tensors with the base two exponential of the elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The base two exponential of the elements. The output has the
            same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import exp2
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = exp2(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.exp2)


def expm1(data: Any) -> Any:
    r"""Return new tensors with the exponential of the elements minus 1.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The exponential of the elements minus 1. The output has the
            same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import expm1
    >>> data = {
    ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    ...     "b": torch.tensor([5, 4, 3, 2, 1]),
    ... }
    >>> out = expm1(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.expm1)


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
