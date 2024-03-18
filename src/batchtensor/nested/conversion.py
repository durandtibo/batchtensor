r"""Contain functions to convert nested data."""

from __future__ import annotations

__all__ = ["from_numpy"]

from typing import Any

import torch

from batchtensor.recursive import recursive_apply


def from_numpy(data: Any) -> Any:
    r"""Create a new nested data structure where the ``numpy.ndarray``s
    are converted to ``torch.Tensor``s.

    Note:
        The returned tensors and ``ndarray``s share the same memory.
        Modifications to the tensor will be reflected in the ndarray
        and vice versa.

    Args:
        data: The input data. Each item must be a tensor.

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
    """
    return recursive_apply(data, torch.from_numpy)
