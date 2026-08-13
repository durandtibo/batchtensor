r"""Implements miscellaneous tensor functions for nested data
structures."""

from __future__ import annotations

__all__ = ["clone", "contiguous", "detach", "pin_memory", "to"]

from typing import TYPE_CHECKING, Any

from coola.recursive import recursive_apply

if TYPE_CHECKING:
    from batchtensor.nested.types import NestedTensor


def clone(data: NestedTensor) -> NestedTensor:
    r"""Return a nested data structure with a copy of each tensor.

    This function recursively applies ``torch.Tensor.clone()`` to all
    tensors in the nested data structure. Unlike the other functions in
    this module, the returned tensors do not share memory with the input
    tensors.

    Args:
        data: The input nested data structure. Can be a dictionary, list,
            tuple, or any combination of these containing tensors. All
            leaf values in the structure must be tensors.

    Returns:
        The data with each tensor replaced by a clone. The structure is
            preserved.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import clone
        >>> data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
        >>> out = clone(data)
        >>> out
        {'a': tensor([[0, 1], [2, 3]]), 'b': tensor([4, 5])}

        ```

    See Also:
        ``batchtensor.nested.detach``: Detach tensors from the computation graph.
    """
    return recursive_apply(data, lambda tensor: tensor.clone())


def contiguous(data: NestedTensor, *args: Any, **kwargs: Any) -> NestedTensor:
    r"""Return a nested data structure with a contiguous copy of each
    tensor.

    This function recursively applies ``torch.Tensor.contiguous()`` to
    all tensors in the nested data structure.

    Args:
        data: The input nested data structure. Can be a dictionary, list,
            tuple, or any combination of these containing tensors. All
            leaf values in the structure must be tensors.
        args: Positional arguments passed to ``torch.Tensor.contiguous``.
        kwargs: Keyword arguments passed to ``torch.Tensor.contiguous``.
            Supports the ``memory_format`` argument.

    Returns:
        The data with each tensor made contiguous. The structure is
            preserved.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import contiguous
        >>> data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
        >>> out = contiguous(data)
        >>> out
        {'a': tensor([[0, 1], [2, 3]]), 'b': tensor([4, 5])}

        ```
    """
    return recursive_apply(data, lambda tensor: tensor.contiguous(*args, **kwargs))


def detach(data: NestedTensor) -> NestedTensor:
    r"""Return a nested data structure with each tensor detached from the
    current computation graph.

    This function recursively applies ``torch.Tensor.detach()`` to all
    tensors in the nested data structure. The returned tensors share
    memory with the input tensors.

    Args:
        data: The input nested data structure. Can be a dictionary, list,
            tuple, or any combination of these containing tensors. All
            leaf values in the structure must be tensors.

    Returns:
        The data with each tensor detached. The structure is preserved.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import detach
        >>> data = {
        ...     "a": torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True),
        ...     "b": torch.tensor([4.0, 5.0]),
        ... }
        >>> out = detach(data)
        >>> out["a"].requires_grad
        False

        ```

    See Also:
        ``batchtensor.nested.clone``: Create an independent copy of each tensor.
    """
    return recursive_apply(data, lambda tensor: tensor.detach())


def pin_memory(data: NestedTensor) -> NestedTensor:
    r"""Return a nested data structure with each tensor copied to pinned
    memory.

    This function recursively applies ``torch.Tensor.pin_memory()`` to
    all tensors in the nested data structure. Pinned (page-locked) CPU
    tensors allow faster host-to-device transfers.

    Args:
        data: The input nested data structure. Can be a dictionary, list,
            tuple, or any combination of these containing tensors. All
            leaf values in the structure must be CPU tensors.

    Returns:
        The data with each tensor copied to pinned memory. The structure
            is preserved.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import pin_memory
        >>> data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
        >>> out = pin_memory(data)  # doctest: +SKIP

        ```
    """
    return recursive_apply(data, lambda tensor: tensor.pin_memory())


def to(data: NestedTensor, *args: Any, **kwargs: Any) -> NestedTensor:
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
