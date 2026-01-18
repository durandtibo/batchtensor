r"""Utility functions for the batchtensor package.

This module provides utility functions that support common tasks when working
with batchtensor, such as random seed management for reproducible experiments.

Available utilities:
    - Random seed management: Functions to get, create, and manage random seeds
      for reproducibility across PyTorch operations.

Example:
    ```pycon
    >>> from batchtensor.utils.seed import get_random_seed
    >>> # Get a random seed for reproducibility
    >>> seed = get_random_seed(42)
    >>> seed
    4224472832458223727

    ```
"""
