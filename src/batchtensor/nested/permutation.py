r"""Implements functions to permute data in nested tensor structures."""

from __future__ import annotations

__all__ = ["permute_along_batch", "permute_along_seq", "shuffle_along_batch", "shuffle_along_seq"]


from functools import partial
from typing import TYPE_CHECKING

from batchtensor import tensor
from batchtensor.constants import BATCH_DIM, SEQ_DIM
from batchtensor.utils.seed import random_permutation
from batchtensor.utils.structure import apply_to_tensors, get_size

if TYPE_CHECKING:
    import torch

    from batchtensor.nested.types import T


def permute_along_batch(data: T, permutation: torch.Tensor) -> T:
    r"""Permute all the tensors along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        permutation: The 1-D tensor containing the indices of the
            permutation. The shape should match the batch dimension
            of the tensor.

    Returns:
        The data with permuted tensors along the batch dimension.
            The output data has the same structure as the input data.

    Raises:
        ValueError: if the permutation is not a 1-D tensor, or if its
            shape does not match the batch dimension of the tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import permute_along_batch
        >>> data = {
        ...     "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        ...     "b": torch.tensor([4, 3, 2, 1, 0]),
        ... }
        >>> out = permute_along_batch(data, torch.tensor([2, 1, 3, 0, 4]))
        >>> out
        {'a': tensor([[4, 5], [2, 3], [6, 7], [0, 1], [8, 9]]), 'b': tensor([2, 3, 1, 4, 0])}

        ```
    """
    return apply_to_tensors(data, partial(tensor.permute_along_batch, permutation=permutation))


def permute_along_seq(data: T, permutation: torch.Tensor) -> T:
    r"""Permute all the tensors along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        permutation: The 1-D tensor containing the indices of the
            permutation. The shape should match the sequence dimension
            of the tensor.

    Returns:
        The data with permuted tensors along the sequence dimension.
            The output data has the same structure as the input data.

    Raises:
        ValueError: if the permutation is not a 1-D tensor, or if its
            shape does not match the sequence dimension of the tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import permute_along_seq
        >>> data = {
        ...     "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": torch.tensor([[4, 3, 2, 1, 0]]),
        ... }
        >>> out = permute_along_seq(data, torch.tensor([2, 1, 3, 0, 4]))
        >>> out
        {'a': tensor([[2, 1, 3, 0, 4], [7, 6, 8, 5, 9]]), 'b': tensor([[2, 3, 1, 4, 0]])}

        ```
    """
    return apply_to_tensors(data, partial(tensor.permute_along_seq, permutation=permutation))


def shuffle_along_batch(data: T, *, generator: torch.Generator | int | None = None) -> T:
    r"""Shuffle all the tensors along the batch dimension.

    Note:
        This function assumes the batch dimension is the first
            dimension of the tensors. All the tensors should have the
            same batch size.

    Args:
        data: The input data. Each item must be a tensor.
        generator: An optional random number generator or random seed.
            All the tensors use the same permutation.

    Returns:
        The data with shuffled tensors along the batch dimension.
            The output data has the same structure as the input data.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import shuffle_along_batch
        >>> data = {
        ...     "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        ...     "b": torch.tensor([4, 3, 2, 1, 0]),
        ... }
        >>> out = shuffle_along_batch(data)
        >>> out
        {'a': tensor([[...]]), 'b': tensor([...])}

        ```
    """
    size = get_size(data, BATCH_DIM)
    return permute_along_batch(
        data=data,
        permutation=random_permutation(size, generator=generator),
    )


def shuffle_along_seq(data: T, *, generator: torch.Generator | int | None = None) -> T:
    r"""Shuffle all the tensors along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension of the tensors. All the tensors should have the
            same sequence size.

    Args:
        data: The input data. Each item must be a tensor.
        generator: An optional random number generator or random seed.
            All the tensors use the same permutation.

    Returns:
        The data with shuffled tensors along the sequence dimension.
            The output data has the same structure as the input data.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.nested import shuffle_along_seq
        >>> data = {
        ...     "a": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": torch.tensor([[4, 3, 2, 1, 0]]),
        ... }
        >>> out = shuffle_along_seq(data)
        >>> out
        {'a': tensor([[...]]), 'b': tensor([[...]])}

        ```
    """
    size = get_size(data, SEQ_DIM)
    return permute_along_seq(
        data=data,
        permutation=random_permutation(size, generator=generator),
    )
