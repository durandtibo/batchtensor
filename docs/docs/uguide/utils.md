# Utility Functions

The `batchtensor.utils` module provides utility functions that support the main tensor operations,
particularly for managing random seeds and ensuring reproducibility.

## Overview

The utils module currently contains seed management utilities that are essential for:

- Generating reproducible random operations
- Managing PyTorch random number generators
- Ensuring consistent behavior across different runs

## Seed Management

### Random Seed Generation

Generate a random seed for reproducible operations:

```pycon
>>> from batchtensor.utils.seed import get_random_seed
>>> seed = get_random_seed(42)
>>> seed
6176747449835261347
>>> # Same input always produces same output
>>> get_random_seed(42)
6176747449835261347
>>> # Different input produces different output
>>> get_random_seed(100)
-6247676604327179579

```

The `get_random_seed` function is useful when you need to derive additional random seeds from a
master seed while maintaining reproducibility.

**Key Features:**

- Returns values between `-2^63` and `2^63 - 1`
- Deterministic: same input always produces same output
- Useful for creating multiple independent random streams

### PyTorch Generator Creation

Create a PyTorch generator with a specific seed:

```pycon
>>> import torch
>>> from batchtensor.utils.seed import get_torch_generator
>>> generator = get_torch_generator(42)
>>> generator
<torch._C.Generator object at 0x...>
>>> # Use with PyTorch random operations
>>> torch.rand(2, 3, generator=generator)
tensor([[0.8823, 0.9150, 0.3829],
        [0.9593, 0.3904, 0.6009]])
>>> # Create a new generator with the same seed for reproducibility
>>> generator = get_torch_generator(42)
>>> torch.rand(2, 3, generator=generator)
tensor([[0.8823, 0.9150, 0.3829],
        [0.9593, 0.3904, 0.6009]])

```

**Parameters:**

- `random_seed` (int): The random seed to initialize the generator
- `device` (torch.device | str | None): The device for the generator (default: "cpu")

**Use Cases:**

- Creating reproducible random tensors
- Ensuring consistent shuffling operations
- Testing and debugging with deterministic behavior

### Generator Setup

Set up a generator from either a seed or an existing generator:

```pycon
>>> import torch
>>> from batchtensor.utils.seed import setup_torch_generator
>>> # From a seed
>>> generator = setup_torch_generator(42)
>>> generator
<torch._C.Generator object at 0x...>
>>> # From an existing generator (returns the same generator)
>>> existing_generator = torch.Generator()
>>> existing_generator.manual_seed(100)
<torch._C.Generator object at 0x...>
>>> result = setup_torch_generator(existing_generator)
>>> result is existing_generator
True

```

This function is particularly useful in library code where you want to accept either a seed or a
generator as input.

## Practical Examples

### Reproducible Shuffling

Use generators to ensure reproducible shuffling:

```pycon
>>> import torch
>>> from batchtensor.tensor import shuffle_along_batch
>>> from batchtensor.utils.seed import get_torch_generator
>>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> # First run with seed 42
>>> generator = get_torch_generator(42)
>>> shuffled = shuffle_along_batch(data, generator=generator)
>>> shuffled
tensor([[7, 8],
        [1, 2],
        [3, 4],
        [5, 6]])
>>> # Second run with same seed produces same result
>>> generator = get_torch_generator(42)
>>> shuffled = shuffle_along_batch(data, generator=generator)
>>> shuffled
tensor([[7, 8],
        [1, 2],
        [3, 4],
        [5, 6]])

```

### Multiple Independent Random Streams

Create multiple independent random streams from a master seed:

```pycon
>>> from batchtensor.utils.seed import get_random_seed, get_torch_generator
>>> master_seed = 42
>>> # Create seeds for different purposes
>>> train_seed = get_random_seed(master_seed)
>>> val_seed = get_random_seed(master_seed + 1)
>>> test_seed = get_random_seed(master_seed + 2)
>>> # Create independent generators
>>> train_gen = get_torch_generator(train_seed)
>>> val_gen = get_torch_generator(val_seed)
>>> test_gen = get_torch_generator(test_seed)

```

### Device-Specific Generators

Create generators for different devices:

```pycon
>>> from batchtensor.utils.seed import get_torch_generator
>>> # CPU generator
>>> cpu_gen = get_torch_generator(42, device="cpu")
>>> # GPU generator (if CUDA is available)
>>> if torch.cuda.is_available():  # doctest: +SKIP
...     gpu_gen = get_torch_generator(42, device="cuda")

```

## Best Practices

1. **Always Use Seeds in Tests**: When writing tests, always use seeded generators to ensure
   reproducible results
2. **Document Random Behavior**: When using random operations in your code, document the expected
   behavior and how to control it with seeds
3. **Separate Random Streams**: Use different seeds for different purposes (training, validation,
   testing) to avoid correlations
4. **Generator Reuse**: Reuse generators when you want to maintain a sequence of random operations,
   create new ones when you want independence

## Thread Safety

Note that PyTorch generators are not thread-safe. If you're using multi-threading, each thread
should have its own generator instance.

## See Also

- [Tensor Operations](tensor.md) - Operations that use random number generators
- [Nested Operations](nested.md) - Nested operations with randomness
- [PyTorch Random Sampling](https://pytorch.org/docs/stable/random.html) - PyTorch's random number
  generation documentation
