r"""Defines important constants for batch and sequence dimensions.

This module provides standardized dimension indices used throughout the
batchtensor package. These constants ensure consistent dimension handling
across all tensor operations.

Constants:
    BATCH_DIM: The batch dimension index (0). This is the first dimension
        of tensors and represents independent samples in a batch. For example,
        in a tensor of shape (batch_size, seq_len, feature_dim), this refers
        to the batch_size dimension.
    SEQ_DIM: The sequence dimension index (1). This is the second dimension
        of tensors and represents sequential data within each batch item.
        For example, in a tensor of shape (batch_size, seq_len, feature_dim),
        this refers to the seq_len dimension.

Example:
    ```pycon
    >>> from batchtensor.constants import BATCH_DIM, SEQ_DIM
    >>> import torch
    >>> # Create a batch of 3 sequences, each of length 5
    >>> tensor = torch.randn(3, 5, 10)  # (batch, seq, features)
    >>> # BATCH_DIM=0 refers to the dimension with size 3
    >>> # SEQ_DIM=1 refers to the dimension with size 5
    >>> tensor.shape[BATCH_DIM]
    3
    >>> tensor.shape[SEQ_DIM]
    5

    ```
"""

from __future__ import annotations

__all__ = ["BATCH_DIM", "SEQ_DIM"]

BATCH_DIM = 0
"""int: The index of the batch dimension in tensors.

This constant is used throughout batchtensor to identify the batch dimension,
which is always assumed to be the first dimension (index 0) of tensors.
"""

SEQ_DIM = 1
"""int: The index of the sequence dimension in tensors.

This constant is used throughout batchtensor to identify the sequence dimension,
which is always assumed to be the second dimension (index 1) of tensors.
"""
