r"""Implements functions to convert between nested data
representations."""

from __future__ import annotations

__all__ = ["as_tensor", "from_numpy", "to_numpy"]

from functools import partial
from typing import Any

import torch
from coola.recursive import recursive_apply


def as_tensor(
    data: Any, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Any:
    r"""Create a new nested data structure with ``torch.Tensor``s.

    This function recursively converts all array-like data (lists, tuples,
    numpy arrays, scalars, or existing tensors) to PyTorch tensors within
    a nested structure. Unlike ``torch.tensor()``, this function shares
    memory with the input data when possible.

    Note:
        This function preserves the structure of the input data while
        converting all array-like values to tensors. The output structure
        mirrors the input structure.

    Args:
        data: The input data. Each item must be a value compatible with
            ``torch.as_tensor``, such as lists, tuples, numpy arrays,
            Python scalars, or existing tensors. Can be a nested structure
            of dictionaries, lists, tuples containing such values.
        dtype: The desired data type of returned tensors. If ``None``,
            it infers data type from the input data. Common dtypes include
            ``torch.float32``, ``torch.int64``, ``torch.bool``, etc.
        device: The device of the constructed tensors (e.g.,
            ``torch.device('cuda')`` or ``'cpu'``). If ``None`` and data
            contains a tensor, the device of that tensor is used. If ``None``
            and data is not a tensor, the result tensor is constructed on
            the CPU.

    Returns:
        A nested data structure with ``torch.Tensor``s. The output data
            has the same structure as the input, with all array-like values
            converted to tensors.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batchtensor.nested import as_tensor
        >>> # Convert mixed types in nested structure
        >>> data = {"a": np.ones((2, 5), dtype=np.float32), "b": np.arange(5), "c": 42}
        >>> out = as_tensor(data)
        >>> out
        {'a': tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]),
         'b': tensor([0, 1, 2, 3, 4]),
         'c': tensor(42)}
        >>> # Specify dtype for all tensors
        >>> out = as_tensor({"values": np.array([1, 2, 3])}, dtype=torch.float32)
        >>> out
        {'values': tensor([1., 2., 3.])}

        ```

    See Also:
        ``batchtensor.nested.from_numpy``: Convert numpy arrays to tensors (shares memory).
        ``batchtensor.nested.to``: Convert tensor dtype/device for existing tensors.
    """
    return recursive_apply(data, partial(torch.as_tensor, dtype=dtype, device=device))


def from_numpy(data: Any) -> Any:
    r"""Create a new nested data structure where the ``numpy.ndarray``s
    are converted to ``torch.Tensor``s.

    This function recursively converts all numpy arrays in a nested structure
    to PyTorch tensors. The conversion uses ``torch.from_numpy()``, which
    creates tensors that share memory with the original arrays.

    Note:
        The returned ``torch.Tensor``s and ``numpy.ndarray``s share the
        same underlying memory. Modifications to the ``torch.Tensor``s will be
        reflected in the ``numpy.ndarray``s and vice versa. This is efficient
        but requires caution when modifying data.

    Warning:
        Since memory is shared, modifying the tensor will modify the original
        numpy array. If you need independent copies, use ``torch.tensor()``
        or call ``.clone()`` on the result.

    Args:
        data: The input data. Each item should be a ``numpy.ndarray``. Can be
            a nested structure of dictionaries, lists, tuples containing numpy
            arrays.

    Returns:
        A nested data structure with ``torch.Tensor``s instead of
            ``numpy.ndarray``s. The output data has the same structure
            as the input, and the tensors share memory with the original
            arrays.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batchtensor.nested import from_numpy
        >>> data = {"a": np.ones((2, 5), dtype=np.float32), "b": np.arange(5)}
        >>> out = from_numpy(data)
        >>> out
        {'a': tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]), 'b': tensor([0, 1, 2, 3, 4])}
        >>> # Demonstrate memory sharing
        >>> data["a"][0, 0] = 999
        >>> out["a"][0, 0]  # Reflects the change
        tensor(999.)

        ```

    See Also:
        ``batchtensor.nested.to_numpy``: Convert tensors to numpy arrays (shares memory).
        ``batchtensor.nested.as_tensor``: More flexible conversion supporting various input types.
    """
    return recursive_apply(data, torch.from_numpy)


def to_numpy(data: Any) -> Any:
    r"""Create a new nested data structure where the ``torch.Tensor``s
    are converted to ``numpy.ndarray``s.

    This function recursively converts all PyTorch tensors in a nested structure
    to numpy arrays. The conversion uses ``tensor.numpy()``, which creates
    arrays that share memory with the original tensors when possible (for
    tensors on CPU with compatible dtypes).

    Note:
        The returned ``torch.Tensor``s and ``numpy.ndarray``s share the
        same underlying memory when tensors are on CPU with compatible dtypes.
        Modifications to the ``torch.Tensor``s will be reflected in the
        ``numpy.ndarray``s and vice versa.

    Warning:
        - Tensors on CUDA devices will be copied to CPU before conversion.
        - Tensors with gradients enabled will raise an error unless you call
          ``.detach()`` first.
        - Since memory is shared (for CPU tensors), modifying the array will
          modify the original tensor.

    Args:
        data: The input data. Each item must be a ``torch.Tensor``. Can be
            a nested structure of dictionaries, lists, tuples containing
            tensors. All tensors should be on CPU without gradients for
            efficient conversion.

    Returns:
        A nested data structure with ``numpy.ndarray``s instead of
            ``torch.Tensor``s. The output data has the same structure
            as the input. For CPU tensors with compatible dtypes, the
            arrays share memory with the original tensors.

    Example:
        ```pycon

        >>> import torch
        >>> from batchtensor.nested import to_numpy
        >>> data = {"a": torch.ones(2, 5), "b": torch.tensor([0, 1, 2, 3, 4])}
        >>> out = to_numpy(data)
        >>> out
        {'a': array([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]], dtype=float32), 'b': array([0, 1, 2, 3, 4])}
        >>> # Demonstrate memory sharing for CPU tensors
        >>> data["a"][0, 0] = 999
        >>> out["a"][0, 0]  # Reflects the change
        np.float32(999.0)

        ```

    See Also:
        ``batchtensor.nested.from_numpy``: Convert numpy arrays to tensors (shares memory).
        ``batchtensor.nested.as_tensor``: Convert various data types to tensors.
    """
    return recursive_apply(data, lambda tensor: tensor.numpy())
