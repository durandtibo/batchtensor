r"""Implements point-wise tensor functions for nested data
structures."""

from __future__ import annotations

__all__ = [
    "abs",
    "clamp",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
]

from functools import partial
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from batchtensor.nested.types import T


from batchtensor.utils.structure import apply_to_tensors


def abs(data: T) -> T:  # noqa: A001
    r"""Return new tensors with the absolute value of each element.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The absolute value of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import abs
        >>> data = {
        ...     "a": torch.tensor([[-4, -3], [-2, -1], [0, 1], [2, 3], [4, 5]]),
        ...     "b": torch.tensor([2, 1, 0, -1, -2]),
        ... }
        >>> out = abs(data)
        >>> out
        {'a': tensor([[4, 3], [2, 1], [0, 1], [2, 3], [4, 5]]), 'b': tensor([2, 1, 0, 1, 2])}

        ```
    """
    return apply_to_tensors(data, torch.abs)


def clamp(
    data: T,
    min: float | None = None,  # noqa: A002
    max: float | None = None,  # noqa: A002
) -> T:
    r"""Clamp all elements in input into the range ``[min, max]``.

    Args:
        data: The input data. Each item must be a tensor.
        min: The lower-bound of the range to be clamped to.
        max: The upper-bound of the range to be clamped to.

    Returns:
        The clamp value of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import clamp
        >>> data = {
        ...     "a": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": torch.tensor([5, 4, 3, 2, 1]),
        ... }
        >>> out = clamp(data, min=1, max=5)
        >>> out
        {'a': tensor([[1, 2], [3, 4], [5, 5], [5, 5], [5, 5]]), 'b': tensor([5, 4, 3, 2, 1])}

        ```
    """
    return apply_to_tensors(data, partial(torch.clamp, min=min, max=max))


def exp(data: T) -> T:
    r"""Return new tensors with the exponential of the elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The exponential of the elements. The output has the same
            structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.exp)


def exp2(data: T) -> T:
    r"""Return new tensors with the base two exponential of the elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The base two exponential of the elements. The output has the
            same structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.exp2)


def expm1(data: T) -> T:
    r"""Return new tensors with the exponential of the elements minus 1.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The exponential of the elements minus 1. The output has the
            same structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.expm1)


def log(data: T) -> T:
    r"""Return new tensors with the natural logarithm of the elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The natural logarithm of the elements. The output has the same
            structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.log)


def log2(data: T) -> T:
    r"""Return new tensors with the logarithm to the base 2 of the
    elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The logarithm to the base 2 of the elements. The output has
            the same structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.log2)


def log10(data: T) -> T:
    r"""Return new tensors with the logarithm to the base 10 of the
    elements.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The logarithm to the base 10 of the elements. The
            output has the same structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.log10)


def log1p(data: T) -> T:
    r"""Return new tensors with the natural logarithm of ``(1 + input)``.

    Args:
        data: The input data. Each item must be a tensor.

    Returns:
        The natural logarithm of ``(1 + input)``. The output has the
            same structure as the input.

    Example:
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
    return apply_to_tensors(data, torch.log1p)
