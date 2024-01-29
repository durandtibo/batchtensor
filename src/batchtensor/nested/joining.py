r"""Contain some tensor joining functions for nested data."""

from __future__ import annotations

__all__ = ["cat_along_batch", "cat_along_seq"]

from typing import TYPE_CHECKING

from batchtensor import tensor

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch


def cat_along_batch(data: Sequence[dict[Hashable, torch.Tensor]]) -> dict[Hashable, torch.Tensor]:
    r"""Concatenate the given tensors in the batch dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data to concatenate. The dictionaries must have
            the same keys.

    Returns:
        The concatenated tensors along the batch dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import cat_along_batch
    >>> data = [
    ...     {
    ...         "a": torch.tensor([[0, 1, 2], [4, 5, 6]]),
    ...         "b": torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ...     },
    ...     {"a": torch.tensor([[7, 8, 9]]), "b": torch.tensor([[17, 18, 19]])},
    ... ]
    >>> out = cat_along_batch(data)
    >>> out
    {'a': tensor([[0, 1, 2], [4, 5, 6], [7, 8, 9]]),
     'b': tensor([[10, 11, 12], [13, 14, 15], [17, 18, 19]])}

    ```
    """
    if not data:
        return {}
    item = data[0]
    return type(item)({key: tensor.cat_along_batch([d[key] for d in data]) for key in item})


def cat_along_seq(data: Sequence[dict[Hashable, torch.Tensor]]) -> dict[Hashable, torch.Tensor]:
    r"""Concatenate the given tensors in the sequence dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data to concatenate. The dictionaries must have
            the same keys.

    Returns:
        The concatenated tensors along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from batchtensor.nested import cat_along_seq
    >>> data = [
    ...     {
    ...         "a": torch.tensor([[0, 1, 2], [4, 5, 6]]),
    ...         "b": torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ...     },
    ...     {"a": torch.tensor([[7], [8]]), "b": torch.tensor([[17], [18]])},
    ... ]
    >>> out = cat_along_seq(data)
    >>> out
    {'a': tensor([[0, 1, 2, 7], [4, 5, 6, 8]]),
     'b': tensor([[10, 11, 12, 17], [13, 14, 15, 18]])}

    ```
    """
    if not data:
        return {}
    item = data[0]
    return type(item)({key: tensor.cat_along_seq([d[key] for d in data]) for key in item})
