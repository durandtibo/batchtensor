r"""Implements miscellaneous tensor functions for nested data
structures."""

from __future__ import annotations

__all__ = ["to"]

from typing import Any

from coola.recursive import recursive_apply


def to(data: Any, *args: Any, **kwargs: Any) -> Any:
    r"""Perform Tensor dtype and/or device conversion.

    Args:
        data: The input data. Each item must be a tensor.
        args: Positional arguments for ``torch.Tensor.to``.
        kwargs: Keyword arguments for ``torch.Tensor.to``.

    Returns:
        The data after conversion.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import to
        >>> data = {
        ...     "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        ...     "b": torch.tensor([4, 3, 2, 1, 0]),
        ... }
        >>> out = to(data, dtype=torch.float)
        >>> out
        {'a': tensor([[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]]),
         'b': tensor([4., 3., 2., 1., 0.])}

        ```
    """
    return recursive_apply(data, lambda tensor: tensor.to(*args, **kwargs))
