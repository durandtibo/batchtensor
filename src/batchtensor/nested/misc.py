r"""Implements miscellaneous tensor functions for nested data
structures."""

from __future__ import annotations

__all__ = ["to"]

from typing import Any

from coola.recursive import recursive_apply


def to(data: Any, *args: Any, **kwargs: Any) -> Any:
    r"""Perform Tensor dtype and/or device conversion on all tensors in
    nested data.

    This function recursively applies ``torch.Tensor.to()`` to all tensors
    in the nested data structure, allowing you to convert dtypes, move to
    different devices, or change other tensor properties for all tensors
    at once.

    Note:
        This function preserves the structure of the input data while
        converting all tensors within it.

    Args:
        data: The input nested data structure. Can be a dictionary, list,
            tuple, or any combination of these containing tensors. All
            leaf values in the structure must be tensors.
        args: Positional arguments passed to ``torch.Tensor.to``. Common
            usage includes passing a device (e.g., ``torch.device('cuda')``),
            dtype (e.g., ``torch.float32``), or another tensor to match
            device and dtype.
        kwargs: Keyword arguments passed to ``torch.Tensor.to``. Supports
            arguments like ``dtype``, ``device``, ``non_blocking``, ``copy``,
            and ``memory_format``.

    Returns:
        The data after conversion. The structure is preserved, with all
            tensors converted according to the specified arguments.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import to
        >>> data = {
        ...     "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        ...     "b": torch.tensor([4, 3, 2, 1, 0]),
        ... }
        >>> # Convert to float dtype
        >>> out = to(data, dtype=torch.float)
        >>> out
        {'a': tensor([[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]]),
         'b': tensor([4., 3., 2., 1., 0.])}
        >>> # Move to GPU (if available) with float32 dtype
        >>> # out = to(data, device='cuda', dtype=torch.float32)

        ```

    See Also:
        ``batchtensor.nested.as_tensor``: Convert data to tensor format.
        ``batchtensor.nested.from_numpy``: Convert numpy arrays to tensors.
    """
    return recursive_apply(data, lambda tensor: tensor.to(*args, **kwargs))
