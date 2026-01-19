r"""Functions for manipulating individual PyTorch tensors with batch and sequence dimensions.

This module provides a collection of functions for working with PyTorch tensors
where the first dimension (index 0) is the batch dimension, and optionally the
second dimension (index 1) is the sequence dimension.

All functions in this module follow these conventions:
    - Functions ending with ``_along_batch`` operate on the batch dimension (dim=0)
    - Functions ending with ``_along_seq`` operate on the sequence dimension (dim=1)
    - The batch dimension always represents independent samples
    - The sequence dimension represents sequential/temporal data within each sample

Function Categories:
    **Reduction operations**: Aggregate values along a dimension
        - ``sum_along_batch``, ``mean_along_batch``, ``max_along_batch``, etc.
        - ``sum_along_seq``, ``mean_along_seq``, ``max_along_seq``, etc.

    **Slicing operations**: Extract subsets of data
        - ``slice_along_batch``, ``select_along_batch``, ``chunk_along_batch``
        - ``slice_along_seq``, ``select_along_seq``, ``chunk_along_seq``

    **Joining operations**: Combine multiple tensors
        - ``cat_along_batch``, ``cat_along_seq``, ``repeat_along_seq``

    **Permutation operations**: Reorder elements
        - ``permute_along_batch``, ``shuffle_along_batch``
        - ``permute_along_seq``, ``shuffle_along_seq``

    **Indexing operations**: Select specific elements
        - ``index_select_along_batch``, ``index_select_along_seq``

    **Comparison operations**: Sort and find extrema
        - ``sort_along_batch``, ``argsort_along_batch``
        - ``sort_along_seq``, ``argsort_along_seq``

    **Mathematical operations**: Cumulative operations
        - ``cumsum_along_batch``, ``cumprod_along_batch``
        - ``cumsum_along_seq``, ``cumprod_along_seq``

Example:
    ```pycon
    >>> import torch
    >>> from batchtensor import tensor as bt
    >>> # Create a batch of 5 samples, each with 2 features
    >>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    >>> # Compute mean along batch dimension (result has shape matching feature dim)
    >>> bt.mean_along_batch(data)
    tensor([5., 6.])
    >>> # Slice first 3 samples from the batch
    >>> bt.slice_along_batch(data, stop=3)
    tensor([[1, 2],
            [3, 4],
            [5, 6]])

    ```

See Also:
    ``batchtensor.nested``: Similar functions for nested data structures containing tensors.
"""

from __future__ import annotations

__all__ = [
    "amax_along_batch",
    "amax_along_seq",
    "amin_along_batch",
    "amin_along_seq",
    "argmax_along_batch",
    "argmax_along_seq",
    "argmin_along_batch",
    "argmin_along_seq",
    "argsort_along_batch",
    "argsort_along_seq",
    "cat_along_batch",
    "cat_along_seq",
    "chunk_along_batch",
    "chunk_along_seq",
    "cumprod_along_batch",
    "cumprod_along_seq",
    "cumsum_along_batch",
    "cumsum_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "max_along_batch",
    "max_along_seq",
    "mean_along_batch",
    "mean_along_seq",
    "median_along_batch",
    "median_along_seq",
    "min_along_batch",
    "min_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "prod_along_batch",
    "prod_along_seq",
    "repeat_along_seq",
    "select_along_batch",
    "select_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
    "slice_along_batch",
    "slice_along_seq",
    "sort_along_batch",
    "sort_along_seq",
    "split_along_batch",
    "split_along_seq",
    "sum_along_batch",
    "sum_along_seq",
]

from batchtensor.tensor.comparison import (
    argsort_along_batch,
    argsort_along_seq,
    sort_along_batch,
    sort_along_seq,
)
from batchtensor.tensor.indexing import index_select_along_batch, index_select_along_seq
from batchtensor.tensor.joining import cat_along_batch, cat_along_seq, repeat_along_seq
from batchtensor.tensor.math import (
    cumprod_along_batch,
    cumprod_along_seq,
    cumsum_along_batch,
    cumsum_along_seq,
)
from batchtensor.tensor.permutation import (
    permute_along_batch,
    permute_along_seq,
    shuffle_along_batch,
    shuffle_along_seq,
)
from batchtensor.tensor.reduction import (
    amax_along_batch,
    amax_along_seq,
    amin_along_batch,
    amin_along_seq,
    argmax_along_batch,
    argmax_along_seq,
    argmin_along_batch,
    argmin_along_seq,
    max_along_batch,
    max_along_seq,
    mean_along_batch,
    mean_along_seq,
    median_along_batch,
    median_along_seq,
    min_along_batch,
    min_along_seq,
    prod_along_batch,
    prod_along_seq,
    sum_along_batch,
    sum_along_seq,
)
from batchtensor.tensor.slicing import (
    chunk_along_batch,
    chunk_along_seq,
    select_along_batch,
    select_along_seq,
    slice_along_batch,
    slice_along_seq,
    split_along_batch,
    split_along_seq,
)
