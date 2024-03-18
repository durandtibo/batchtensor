r"""Contain functions to convert nested data."""

from __future__ import annotations

__all__ = ["from_numpy", "to_numpy"]

from typing import Any

import torch

from batchtensor.recursive import recursive_apply


def from_numpy(data: Any) -> Any:
    r"""Create a new nested data structure where the ``numpy.ndarray``s
    are converted to ``torch.Tensor``s.

    Note:
        The returned ``torch.Tensor``s and ``numpy.ndarray``s share the
        same memory. Modifications to the ``torch.Tensor``s will be
        reflected in the ``numpy.ndarray``s and vice versa.

    Args:
        data: The input data. Each item must be a ``torch.Tensor``.

    Returns:
        A nested data structure with ``torch.Tensor``s instead of
            ``numpy.ndarray``s. The output data has the same structure
            as the input.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from batchtensor.nested import from_numpy
    >>> data = {"a": np.ones((2, 5), dtype=np.float32), "b": np.arange(5)}
    >>> out = from_numpy(data)
    >>> out
    {'a': tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]), 'b': tensor([0, 1, 2, 3, 4])}

    ```
    """
    return recursive_apply(data, torch.from_numpy)


def to_numpy(data: Any) -> Any:
    r"""Create a new nested data structure where the ``torch.Tensor``s
    are converted to ``numpy.ndarray``s.

    Note:
        The returned ``torch.Tensor``s and ``numpy.ndarray``s share the
        same memory. Modifications to the ``torch.Tensor``s will be
        reflected in the ``numpy.ndarray``s and vice versa.

    Args:
        data: The input data. Each item must be a ``numpy.ndarray``.

    Returns:
        A nested data structure with ``numpy.ndarray``s instead of
            ``torch.Tensor``s. The output data has the same structure
            as the input.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from batchtensor.nested import to_numpy
    >>> data = {"a": torch.ones(2, 5), "b": torch.arange(5)}
    >>> out = to_numpy(data)
    >>> out
    {'a': array([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]], dtype=float32), 'b': array([0, 1, 2, 3, 4])}

    ```
    """
    return recursive_apply(data, lambda tensor: tensor.numpy())
