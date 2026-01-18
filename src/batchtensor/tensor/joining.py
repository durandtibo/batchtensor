r"""Implements joining functions for tensors."""

from __future__ import annotations

__all__ = ["cat_along_batch", "cat_along_seq", "repeat_along_seq"]


import torch

from batchtensor.constants import BATCH_DIM, SEQ_DIM


def cat_along_batch(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    r"""Concatenate the given tensors in the batch dimension.

    This function concatenates multiple tensors along their batch dimension,
    stacking batch items from all input tensors into a single tensor. The
    batch dimension sizes can differ, but all other dimensions must match.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the batch dimension is the first
            dimension (index 0).

    Args:
        tensors: A sequence (list or tuple) of tensors to concatenate. All
            tensors must have the same number of dimensions and the same
            shape except in the batch dimension. At least one tensor must
            be provided.

    Returns:
        The concatenated tensor along the batch dimension. If the input
            tensors have batch sizes ``[b1, b2, ..., bn]``, the output will
            have batch size ``b1 + b2 + ... + bn``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import cat_along_batch
        >>> tensors = [
        ...     torch.tensor([[0, 1, 2], [4, 5, 6]]),
        ...     torch.tensor([[10, 11, 12], [13, 14, 15]]),
        ... ]
        >>> out = cat_along_batch(tensors)
        >>> out
        tensor([[ 0,  1,  2],
                [ 4,  5,  6],
                [10, 11, 12],
                [13, 14, 15]])
        >>> # Concatenating tensors with different batch sizes
        >>> tensors = [
        ...     torch.tensor([[1, 2]]),  # batch size 1
        ...     torch.tensor([[3, 4], [5, 6], [7, 8]]),  # batch size 3
        ... ]
        >>> out = cat_along_batch(tensors)
        >>> out
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])

        ```

    See Also:
        ``cat_along_seq``: Concatenate along the sequence dimension instead.
        ``split_along_batch``: Inverse operation - split a tensor into chunks.
        ``torch.cat``: PyTorch's general concatenation function.
    """
    return torch.cat(tensors, dim=BATCH_DIM)


def cat_along_seq(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    r"""Concatenate the given tensors in the sequence dimension.

    All tensors must either have the same data type and shape (except
    in the concatenating dimension) or be empty.

    Note:
        This function assumes the sequence dimension is the second
            dimension.

    Args:
        tensors: The tensors to concatenate.

    Returns:
        The concatenated tensors along the sequence dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import cat_along_seq
        >>> tensors = [
        ...     torch.tensor([[0, 1, 2], [4, 5, 6]]),
        ...     torch.tensor([[10, 11], [12, 13]]),
        ... ]
        >>> out = cat_along_seq(tensors)
        >>> out
        tensor([[ 0,  1,  2, 10, 11],
                [ 4,  5,  6, 12, 13]])

        ```
    """
    return torch.cat(tensors, dim=SEQ_DIM)


def repeat_along_seq(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    r"""Repeat the data along the sequence dimension.

    This function repeats the sequence data a specified number of times,
    effectively duplicating the sequence content. The resulting tensor has
    a sequence length that is ``repeats`` times the original sequence length.

    Note:
        This function assumes the sequence dimension is the second
            dimension (index 1).

    Args:
        tensor: The input tensor. Must have at least two dimensions
            (batch and sequence).
        repeats: The number of times to repeat the data along the sequence
            dimension. Must be a positive integer. If ``repeats=1``, returns
            a copy of the input.

    Returns:
        A new tensor with the data repeated along the sequence dimension.
            If the input has shape ``(batch_size, seq_len, ...)``, the output
            will have shape ``(batch_size, seq_len * repeats, ...)``.

    Example:
        ```pycon
        >>> import torch
        >>> from batchtensor.tensor import repeat_along_seq
        >>> tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = repeat_along_seq(tensor, 2)
        >>> out
        tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]])
        >>> # Repeat 3 times
        >>> out = repeat_along_seq(tensor, 3)
        >>> out
        tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9]])

        ```

    See Also:
        ``cat_along_seq``: Concatenate different tensors (not repeating the same).
        ``torch.repeat``: PyTorch's general repeat function for all dimensions.
    """
    sizes = [1] * tensor.dim()
    sizes[1] = repeats
    return tensor.repeat(*sizes)
