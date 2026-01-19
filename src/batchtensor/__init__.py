r"""Batchtensor: A lightweight library for manipulating batches of PyTorch tensors.

This package provides functions for manipulating PyTorch tensors where the first
dimension is the batch dimension, and optionally the second dimension is the
sequence dimension. It also supports nested data structures (dictionaries, lists,
tuples) containing tensors, applying operations recursively to all tensors.

The package is organized into several submodules:

- ``batchtensor.tensor``: Functions for manipulating individual tensors with batch
  and sequence dimensions.
- ``batchtensor.nested``: Functions for manipulating nested data structures
  (dicts, lists, tuples) containing tensors. These functions recursively apply
  tensor operations to all tensors in the structure.
- ``batchtensor.utils``: Utility functions for common tasks like random seed
  management.
- ``batchtensor.constants``: Important constants like BATCH_DIM and SEQ_DIM.

Key Features:
    - Batch operations: All functions assume the first dimension is the batch
      dimension (BATCH_DIM = 0).
    - Sequence operations: Functions with ``_along_seq`` suffix assume the second
      dimension is the sequence dimension (SEQ_DIM = 1).
    - Nested data support: The ``nested`` module allows you to work with complex
      data structures while maintaining the same simple API.
    - Type safety: All functions include type hints for better IDE support and
      static type checking.
    - Comprehensive documentation: Every function includes detailed docstrings
      with working examples.

Example:
    Basic usage with individual tensors:

    ```pycon
    >>> import torch
    >>> from batchtensor.tensor import slice_along_batch
    >>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> slice_along_batch(tensor, stop=3)
    tensor([[0, 1],
            [2, 3],
            [4, 5]])

    ```

    Working with nested data structures:

    ```pycon
    >>> from batchtensor.nested import slice_along_batch
    >>> batch = {
    ...     "features": torch.tensor([[1, 2], [3, 4], [5, 6]]),
    ...     "labels": torch.tensor([0, 1, 0]),
    ... }
    >>> slice_along_batch(batch, stop=2)
    {'features': tensor([[1, 2], [3, 4]]), 'labels': tensor([0, 1])}

    ```

See Also:
    - Documentation: https://durandtibo.github.io/batchtensor/
    - GitHub: https://github.com/durandtibo/batchtensor
"""

from __future__ import annotations

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
