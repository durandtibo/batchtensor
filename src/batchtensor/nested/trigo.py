r"""Contain some tensor trigonometric functions for nested data."""

from __future__ import annotations

__all__ = ["acosh", "asinh", "atanh"]

from typing import Any

import torch

from batchtensor.recursive import recursive_apply


def acos(data: Any) -> Any:
    r"""Return new tensors with the inverse cosine of each element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The inverse cosine of the elements. The output has the same
            structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import acos
    >>> data = {"a": torch.randn(5, 2), "b": torch.rand(5)}
    >>> out = acos(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.acos)


def acosh(data: Any) -> Any:
    r"""Return new tensors with the inverse hyperbolic cosine of each
    element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The inverse hyperbolic cosine of the elements. The output has
            the same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import asinh
    >>> data = {"a": torch.randn(5, 2), "b": torch.rand(5)}
    >>> out = asinh(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.acosh)


def asin(data: Any) -> Any:
    r"""Return new tensors with the arcsine of each element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The arcsine of the elements. The output has the same
            structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import asin
    >>> data = {"a": torch.randn(5, 2), "b": torch.rand(5)}
    >>> out = asin(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.asin)


def asinh(data: Any) -> Any:
    r"""Return new tensors with the inverse hyperbolic sine of each
    element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The inverse hyperbolic sine of the elements. The output has
            the same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import asinh
    >>> data = {"a": torch.randn(5, 2), "b": torch.rand(5)}
    >>> out = asinh(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.asinh)


def atan(data: Any) -> Any:
    r"""Return new tensors with the arctangent of each element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The arctangent of the elements. The output has the same
            structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import atan
    >>> data = {"a": torch.randn(5, 2), "b": torch.rand(5)}
    >>> out = atan(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.atan)


def atanh(data: Any) -> Any:
    r"""Return new tensors with the inverse hyperbolic tangent of each
    element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The inverse hyperbolic tangent of the elements. The output has
            the same structure as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import atanh
    >>> data = {"a": torch.randn(5, 2), "b": torch.rand(5)}
    >>> out = atanh(data)
    >>> out
    {'a': tensor([[...]]), 'b': tensor([...])}

    ```
    """
    return recursive_apply(data, torch.atanh)
