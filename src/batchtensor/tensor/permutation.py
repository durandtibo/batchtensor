r"""Implements functions to permute data in tensors."""

from __future__ import annotations

__all__ = ["permute_along_batch", "permute_along_seq", "shuffle_along_batch", "shuffle_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def permute_along_batch(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    r"""Permute the tensor along the batch dimension.

    This function reorders the elements along the batch dimension according
    to a specified permutation. The permutation defines a mapping from new
    positions to original positions: ``output[i] = input[permutation[i]]``.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The tensor to permute. Must have at least one dimension.
        permutation: A 1-D tensor containing the indices of the permutation.
            Must be a ``LongTensor`` with shape ``(batch_size,)``. Each index
            must be in the range ``[0, batch_size-1]`` and all indices should
            be unique (though duplicates are allowed if you want to repeat
            certain batch items).

    Returns:
        The tensor with permuted data along the batch dimension. The shape
            is unchanged, but elements are reordered according to the
            permutation.

    Raises:
        RuntimeError: if the shape of the permutation does not match
            the batch dimension of the tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import permute_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> # Reverse the batch order
        >>> out = permute_along_batch(tensor, torch.tensor([4, 3, 2, 1, 0]))
        >>> out
        tensor([[8, 9],
                [6, 7],
                [4, 5],
                [2, 3],
                [0, 1]])
        >>> # Custom permutation
        >>> out = permute_along_batch(tensor, torch.tensor([2, 1, 3, 0, 4]))
        >>> out
        tensor([[4, 5],
                [2, 3],
                [6, 7],
                [0, 1],
                [8, 9]])

        ```

    See Also:
        ``shuffle_along_batch``: Apply a random permutation.
        ``permute_along_seq``: Permute the sequence dimension instead.
        ``index_select_along_batch``: Select specific indices (can duplicate or omit items).
    """
    if permutation.shape[0] != tensor.shape[0]:
        msg = (
            f"permutation shape ({permutation.shape}) is not compatible with tensor shape "
            f"({tensor.shape})"
        )
        raise RuntimeError(msg)
    return tensor.index_select(dim=BATCH_DIM, index=permutation)


def permute_along_seq(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    r"""Permute the tensor along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The tensor to split.
        permutation: The 1-D tensor containing the indices of the
            permutation. The shape should match the sequence dimension
            of the tensor.

    Returns:
        The tensor with permuted data along the sequence dimension.

    Raises:
        RuntimeError: if the shape of the permutation does not match
            the sequence dimension of the tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import permute_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = permute_along_seq(tensor, torch.tensor([2, 1, 3, 0, 4]))
        >>> out
        tensor([[2, 1, 3, 0, 4],
                [7, 6, 8, 5, 9]])

        ```
    """
    if permutation.shape[0] != tensor.shape[1]:
        msg = (
            f"permutation shape ({permutation.shape}) is not compatible with tensor shape "
            f"({tensor.shape})"
        )
        raise RuntimeError(msg)
    return tensor.index_select(dim=SEQ_DIM, index=permutation)


def shuffle_along_batch(
    tensor: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Shuffle the tensor along the batch dimension.

    This function randomly reorders the elements along the batch dimension,
    creating a random permutation of the batch items. All elements within
    each batch item are kept together and maintain their relative positions.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensor: The tensor to shuffle. Must have at least one dimension.
        generator: An optional random number generator for reproducible
            shuffling. If provided, the shuffling will be deterministic
            based on the generator's state. If ``None``, uses PyTorch's
            default random state.

    Returns:
        The shuffled tensor. The shape is unchanged, but elements along
            the batch dimension are reordered randomly.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import shuffle_along_batch
        >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = shuffle_along_batch(tensor)
        >>> out  # Order is random
        tensor([[...]])
        >>> # For reproducible shuffling
        >>> generator = torch.Generator().manual_seed(42)
        >>> out = shuffle_along_batch(tensor, generator=generator)
        >>> out
        tensor([[6, 7], [2, 3], [8, 9], [0, 1], [4, 5]])

        ```

    See Also:
        ``permute_along_batch``: Apply a specific permutation (not random).
        ``shuffle_along_seq``: Shuffle the sequence dimension instead.
        ``batchtensor.utils.manual_seed``: Set global random seed for reproducibility.
    """
    return permute_along_batch(
        tensor=tensor,
        permutation=torch.randperm(tensor.shape[BATCH_DIM], generator=generator),
    )


def shuffle_along_seq(
    tensor: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Shuffle the tensor along the sequence dimension.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensor: The tensor to split.
        generator: An optional random number generator.

    Returns:
        The shuffled tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import shuffle_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = shuffle_along_seq(tensor)
        >>> out
        tensor([[...]])

        ```
    """
    return permute_along_seq(
        tensor=tensor,
        permutation=torch.randperm(tensor.shape[SEQ_DIM], generator=generator),
    )
