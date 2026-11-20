# Utility Functions

The `batchtensor.utils` module contains the building blocks of the library. You can use them to
write your own operations on nested data structures, and to draw reproducible permutations.

| Module                        | Content                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| `batchtensor.utils.structure` | `apply_to_tensors`, `split_tensors`, `combine_tensors`, `get_size` |
| `batchtensor.utils.types`     | the `NestedTensor` type and the type variable `T`                  |
| `batchtensor.utils.seed`      | `setup_torch_generator`, `random_permutation`                      |

## Structure Helpers

All the functions of `batchtensor.nested` are built on three helpers, one for each shape of
operation. They are the only code that walks a nested structure, so they define the rules for
containers and leaves:

- The structure is a tensor, or a `dict`, `list` or `tuple` of structures (and their subclasses).
- The output keeps the container types and the keys of the input.
- A leaf that is not a `torch.Tensor` raises a `TypeError`.

| Helper                       | Shape  | Use                                                     |
| ---------------------------- | ------ | ------------------------------------------------------- |
| `apply_to_tensors(data, fn)` | 1 -> 1 | apply `fn` to every leaf and keep the structure         |
| `split_tensors(data, fn)`    | 1 -> N | split every leaf with `fn` and regroup the pieces       |
| `combine_tensors(items, fn)` | N -> 1 | combine N structures with the same layout, leaf by leaf |

`get_size(data, dim)` is a fourth helper: it returns the size along `dim` shared by all the tensors.

### Apply a function to every tensor

Use `apply_to_tensors` for an operation that has no `batchtensor` function, such as an element-wise
function:

```pycon
>>> import torch
>>> from functools import partial
>>> from batchtensor.utils.structure import apply_to_tensors
>>> batch = {"x": torch.tensor([[-1.0, 2.0]]), "y": [torch.tensor([-3.0])]}
>>> apply_to_tensors(batch, torch.abs)
{'x': tensor([[1., 2.]]), 'y': [tensor([3.])]}
>>> apply_to_tensors(batch, partial(torch.clamp, min=0.0))
{'x': tensor([[0., 2.]]), 'y': [tensor([0.])]}

```

### Split every tensor

`fn` returns a tuple of tensors. The i-th output has the structure of the input and the i-th piece
of every leaf. All the leaves must give the same number of pieces:

```pycon
>>> import torch
>>> from batchtensor.utils.structure import split_tensors
>>> data = {"a": torch.arange(4), "b": [torch.arange(4)]}
>>> split_tensors(data, lambda t: t.chunk(2, dim=0))
({'a': tensor([0, 1]), 'b': [tensor([0, 1])]}, {'a': tensor([2, 3]), 'b': [tensor([2, 3])]})

```

### Combine structures

`fn` receives the list of the tensors found at the same position in each structure, and returns one
tensor. The structures must have the same layout (the same keys and lengths):

```pycon
>>> import torch
>>> from batchtensor.utils.structure import combine_tensors
>>> items = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3])}]
>>> combine_tensors(items, torch.cat)
{'a': tensor([1, 2, 3])}

```

### Read a size

```pycon
>>> import torch
>>> from batchtensor.utils.structure import get_size
>>> get_size({"x": torch.ones(4, 3), "y": [torch.zeros(4)]}, dim=0)
4

```

`get_size` raises a `ValueError` if the structure has no tensor, if a tensor has too few
dimensions, or if the sizes differ. `batchtensor.nested.batch_size` and `seq_len` call it with the
batch and sequence dimensions.

## Types

`NestedTensor` is the type of a tensor, or of a mapping, list or tuple of `NestedTensor`.
`batchtensor.nested` re-exports it. `str` and `bytes` are not accepted, and sets and dataclasses are
outside the contract. `T` is a type variable bound to `NestedTensor`: a function typed `T -> T`
returns the type it receives, so a dict in gives a dict out.

```python
from batchtensor.utils.structure import apply_to_tensors
from batchtensor.utils.types import T


def negate(data: T) -> T:
    return apply_to_tensors(data, lambda t: -t)
```

## Random Permutations

`random_permutation(size, generator=None)` returns a 1-D `torch.long` tensor with a random
permutation of `0..size-1`. `generator` is a `torch.Generator`, an `int` seed or `None`, in which
case the global torch random state is used. The permutation is created on the device of the
generator. `setup_torch_generator` turns a seed or a generator into a `torch.Generator`: a
generator is returned as is, and a seed makes a new one.

```pycon
>>> import torch
>>> from batchtensor.utils.seed import random_permutation, setup_torch_generator
>>> bool((random_permutation(5, generator=42) == random_permutation(5, generator=42)).all())
True
>>> generator = torch.Generator().manual_seed(7)
>>> setup_torch_generator(generator) is generator
True

```

The `shuffle_along_*` functions draw one permutation with `random_permutation` and apply it to
every tensor. To control the randomness of your own code, create generators with
[`coola.random`](https://durandtibo.github.io/coola/): `make_torch_generator(seed, device=...)`
creates a seeded `torch.Generator` and `sample_torch_seed(seed)` derives a deterministic seed from a
master seed.

```pycon
>>> import torch
>>> from coola.random import make_torch_generator
>>> from batchtensor.tensor import shuffle_along_batch
>>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> generator = make_torch_generator(42)
>>> shuffled = shuffle_along_batch(data, generator=generator)

```

A generator is stateful: reuse one to continue a sequence of random operations, and create a new one
from the same seed to repeat it.

## See Also

- [Tensor Operations](tensor.md) - Operations that use random number generators
- [Nested Operations](nested.md) - Functions built on the structure helpers
- [Design principles](../design_principles.md) - The rules the helpers define
- [PyTorch Random Sampling](https://pytorch.org/docs/stable/random.html) - PyTorch's random number
  generation documentation
