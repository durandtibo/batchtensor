r"""Implements utility functions to manage random seeds for reproducible
tensor operations."""

from __future__ import annotations

__all__ = ["random_permutation", "setup_torch_generator"]


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


def random_permutation(size: int, generator: torch.Generator | int | None = None) -> torch.Tensor:
    r"""Generate a random permutation of the integers from 0 to ``size - 1``.

    Args:
        size: The number of elements of the permutation.
        generator: A ``torch.Generator`` object, a random seed or
            ``None``. If ``None``, the global random state of PyTorch
            is used. The permutation is created on the device of the
            generator.

    Returns:
        A 1-D ``torch.long`` tensor with the permutation.

    Example:
        ```pycon
        >>> from batchtensor.utils.seed import random_permutation
        >>> random_permutation(5, generator=42)
        tensor([...])

        ```
    """
    if generator is None:
        return torch.randperm(size)
    generator = setup_torch_generator(generator)
    return torch.randperm(size, generator=generator, device=generator.device)
