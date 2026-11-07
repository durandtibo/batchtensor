# Utility Functions

The `batchtensor.utils` module provides utility functions that support the main tensor operations,
particularly for managing random seeds and ensuring reproducibility.

## Overview

The utils module currently contains seed management utilities that are essential for:

- Generating reproducible random operations
- Managing PyTorch random number generators
- Ensuring consistent behavior across different runs

## Seed Management

Seeds and generators are created with [`coola.random`](https://durandtibo.github.io/coola/):
`make_torch_generator(seed, device=...)` creates a seeded `torch.Generator` and
`sample_torch_seed(seed)` derives a deterministic seed from a master seed.

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
>>> from coola.random import make_torch_generator
>>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> # First run with seed 42
>>> generator = make_torch_generator(42)
>>> shuffled = shuffle_along_batch(data, generator=generator)
>>> # Second run with same seed produces same result
>>> generator = make_torch_generator(42)
>>> shuffled = shuffle_along_batch(data, generator=generator)

```

### Multiple Independent Random Streams

Create multiple independent random streams from a master seed:

```pycon
>>> from coola.random import make_torch_generator, sample_torch_seed
>>> master_seed = 42
>>> # Create seeds for different purposes
>>> train_seed = sample_torch_seed(master_seed)
>>> val_seed = sample_torch_seed(master_seed + 1)
>>> test_seed = sample_torch_seed(master_seed + 2)
>>> # Create independent generators
>>> train_gen = make_torch_generator(train_seed)
>>> val_gen = make_torch_generator(val_seed)
>>> test_gen = make_torch_generator(test_seed)

```

### Device-Specific Generators

Create generators for different devices:

```pycon
>>> from coola.random import make_torch_generator
>>> # CPU generator
>>> cpu_gen = make_torch_generator(42, device="cpu")
>>> # GPU generator (if CUDA is available)
>>> if torch.cuda.is_available():  # doctest: +SKIP
...     gpu_gen = make_torch_generator(42, device="cuda")

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

## See Also

- [Tensor Operations](tensor.md) - Operations that use random number generators
- [Nested Operations](nested.md) - Nested operations with randomness
- [PyTorch Random Sampling](https://pytorch.org/docs/stable/random.html) - PyTorch's random number
  generation documentation

## Structure Helpers

`batchtensor.utils.structure` contains the three helpers that all the nested functions use:

- `apply_to_tensors(data, fn)`: apply `fn` to every tensor leaf and keep the structure.
- `split_tensors(data, fn)`: split every tensor with `fn` and regroup the pieces in N structures.
- `combine_tensors(items, fn)`: combine N structures with the same layout, leaf by leaf.
- `get_size(data, dim)`: the size shared by all the tensors along a dimension.

A leaf that is not a tensor raises a `TypeError`.

```pycon
>>> import torch
>>> from batchtensor.utils.structure import apply_to_tensors, get_size
>>> batch = {"x": torch.ones(4, 3), "y": [torch.zeros(4)]}
>>> get_size(batch, dim=0)
4
>>> apply_to_tensors(batch, torch.neg)
{'x': tensor([[-1., -1., -1.],
              [-1., -1., -1.],
              [-1., -1., -1.],
              [-1., -1., -1.]]), 'y': [tensor([-0., -0., -0., -0.])]}

```
