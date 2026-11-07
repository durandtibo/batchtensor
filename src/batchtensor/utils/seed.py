r"""Implements utility functions to manage random seeds for reproducible
tensor operations."""

from __future__ import annotations

__all__ = ["setup_torch_generator"]


import torch
from coola.random import make_torch_generator


def setup_torch_generator(generator_or_seed: int | torch.Generator) -> torch.Generator:
    r"""Set up a ``torch.Generator`` object.

    Args:
        generator_or_seed: A ``torch.Generator`` object or a random
            seed.

    Returns:
        A ``torch.Generator`` object.

    Example:
        ```pycon
        >>> from batchtensor.utils.seed import setup_torch_generator
        >>> generator = setup_torch_generator(42)
        >>> generator
        <torch._C.Generator object at 0x...>

        ```
    """
    if isinstance(generator_or_seed, torch.Generator):
        return generator_or_seed
    return make_torch_generator(generator_or_seed)
