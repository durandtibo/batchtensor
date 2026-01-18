r"""Functions for manipulating nested data structures containing PyTorch tensors.

This module provides functions for working with nested data structures (dictionaries,
lists, tuples) that contain PyTorch tensors. All functions recursively apply
operations to every tensor in the structure while preserving the structure itself.

The module uses the ``coola`` library's recursive application capabilities to
handle arbitrary nesting levels and mixed data structures. This allows you to
work with complex batched data (e.g., a dictionary of lists of tensors) using
the same simple API as individual tensors.

All functions in this module follow these conventions:
    - Functions ending with ``_along_batch`` operate on the batch dimension (dim=0)
    - Functions ending with ``_along_seq`` operate on the sequence dimension (dim=1)
    - All tensors in the nested structure must have compatible shapes for the
      operation being performed
    - The output structure mirrors the input structure

Function Categories:
    **Reduction operations**: Aggregate values along a dimension
        - ``sum_along_batch``, ``mean_along_batch``, ``max_along_batch``, etc.

    **Slicing operations**: Extract subsets of data
        - ``slice_along_batch``, ``select_along_batch``, ``chunk_along_batch``
        - ``slice_along_seq``, ``select_along_seq``, ``chunk_along_seq``

    **Joining operations**: Combine multiple nested structures
        - ``cat_along_batch``, ``cat_along_seq``, ``repeat_along_seq``

    **Permutation operations**: Reorder elements
        - ``permute_along_batch``, ``shuffle_along_batch``
        - ``permute_along_seq``, ``shuffle_along_seq``

    **Indexing operations**: Select specific elements
        - ``index_select_along_batch``, ``index_select_along_seq``

    **Comparison operations**: Sort and find extrema
        - ``sort_along_batch``, ``argsort_along_batch``

    **Mathematical operations**: Element-wise and cumulative operations
        - ``cumsum_along_batch``, ``cumprod_along_batch``
        - ``abs``, ``exp``, ``log``, ``clamp``

    **Trigonometric operations**: Standard trigonometric functions
        - ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``
        - ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``

    **Conversion operations**: Convert between tensor types and formats
        - ``as_tensor``, ``from_numpy``, ``to_numpy``, ``to``

Example:
    Working with nested dictionaries:

    ```pycon
    >>> import torch
    >>> from batchtensor import nested as bn
    >>> # Create a nested batch structure
    >>> batch = {
    ...     "inputs": torch.tensor([[1, 2], [3, 4], [5, 6]]),
    ...     "labels": torch.tensor([0, 1, 0]),
    ...     "metadata": {"ids": torch.tensor([10, 20, 30])},
    ... }
    >>> # Slice the first 2 samples from all tensors
    >>> bn.slice_along_batch(batch, stop=2)
    {'inputs': tensor([[1, 2], [3, 4]]),
     'labels': tensor([0, 1]),
     'metadata': {'ids': tensor([10, 20])}}

    ```

    Working with nested lists:

    ```pycon
    >>> data = [
    ...     torch.tensor([[1, 2], [3, 4]]),
    ...     torch.tensor([[5, 6], [7, 8]]),
    ... ]
    >>> # Apply absolute value to all tensors
    >>> bn.abs(data)
    [tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]])]

    ```

See Also:
    ``batchtensor.tensor``: Similar functions for individual tensors.
"""

from __future__ import annotations

__all__ = [
    "abs",
    "acos",
    "acosh",
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
    "as_tensor",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cat_along_batch",
    "cat_along_seq",
    "chunk_along_batch",
    "chunk_along_seq",
    "clamp",
    "cos",
    "cosh",
    "cumprod_along_batch",
    "cumprod_along_seq",
    "cumsum_along_batch",
    "cumsum_along_seq",
    "exp",
    "exp2",
    "expm1",
    "from_numpy",
    "index_select_along_batch",
    "index_select_along_seq",
    "log",
    "log1p",
    "log2",
    "log10",
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
    "sin",
    "sinh",
    "slice_along_batch",
    "slice_along_seq",
    "sort_along_batch",
    "sort_along_seq",
    "split_along_batch",
    "split_along_seq",
    "sum_along_batch",
    "sum_along_seq",
    "tan",
    "tanh",
    "to",
    "to_numpy",
]

from batchtensor.nested.comparison import (
    argsort_along_batch,
    argsort_along_seq,
    sort_along_batch,
    sort_along_seq,
)
from batchtensor.nested.conversion import as_tensor, from_numpy, to_numpy
from batchtensor.nested.indexing import index_select_along_batch, index_select_along_seq
from batchtensor.nested.joining import cat_along_batch, cat_along_seq, repeat_along_seq
from batchtensor.nested.math import (
    cumprod_along_batch,
    cumprod_along_seq,
    cumsum_along_batch,
    cumsum_along_seq,
)
from batchtensor.nested.misc import to
from batchtensor.nested.permutation import (
    permute_along_batch,
    permute_along_seq,
    shuffle_along_batch,
    shuffle_along_seq,
)
from batchtensor.nested.pointwise import abs  # noqa: A004
from batchtensor.nested.pointwise import (
    clamp,
    exp,
    exp2,
    expm1,
    log,
    log1p,
    log2,
    log10,
)
from batchtensor.nested.reduction import (
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
from batchtensor.nested.slicing import (
    chunk_along_batch,
    chunk_along_seq,
    select_along_batch,
    select_along_seq,
    slice_along_batch,
    slice_along_seq,
    split_along_batch,
    split_along_seq,
)
from batchtensor.nested.trigo import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    sin,
    sinh,
    tan,
    tanh,
)
