r"""Utility functions for the batchtensor package.

This module provides utility functions that support common tasks when working
with batchtensor, such as random seed management for reproducible experiments.

Available utilities:
    - Random seed management: Functions to get, manually set, and automatically
      manage random seeds for reproducibility across PyTorch, NumPy, and Python's
      random module.

Example:
    ```pycon
    >>> from batchtensor.utils import manual_seed
    >>> # Set seed for reproducibility
    >>> manual_seed(42)
    42

    ```
"""
