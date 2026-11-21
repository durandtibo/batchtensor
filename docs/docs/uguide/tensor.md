# Tensor Operations

The `batchtensor.tensor` module provides functions to manipulate individual PyTorch tensors along
batch and sequence dimensions. These functions are the foundation for the nested operations and can
be used directly when working with single tensors.

## Overview

The tensor module operates on individual tensors with two dimensional conventions:

- **Batch dimension**: The first dimension (dimension 0) represents the batch
- **Sequence dimension**: The second dimension (dimension 1) represents the sequence/time steps

All functions in this module follow these conventions. The shape assumptions are:

- Batch operations: `(batch_size, *)` where `*` means any additional dimensions
- Sequence operations: `(batch_size, seq_len, *)` where `*` means any additional dimensions

A function that works on a dimension comes in two variants, `<name>_along_batch` and
`<name>_along_seq`, and takes no `dim` argument. Apart from the functions listed in
[Design principles](../design_principles.md#tensor-functions-are-thin-wrappers-p2), each function is
one torch call with the dimension fixed: it has the semantics of the torch function, and does not
copy, cast or move data.

## Functions by family

| Family      | Functions (each with `_along_batch` and `_along_seq`, unless noted)                                                        |
| ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| Slicing     | `slice`, `select`, `chunk`, `split`                                                                                        |
| Indexing    | `index_select`                                                                                                             |
| Joining     | `cat`, `stack`, `repeat`                                                                                                   |
| Padding     | `pad`, and `pad_sequences` (no variant)                                                                                    |
| Masking     | `lengths_to_mask` (no variant), `masked_sum`, `masked_mean`, `masked_amax`, `masked_amin` (`_along_seq` only)              |
| Reduction   | `sum`, `mean`, `prod`, `amax`, `amin`, `max`, `min`, `argmax`, `argmin`, `median`, `std`, `var`, `any`, `all`, `logsumexp` |
| Ordering    | `sort`, `argsort`, `topk`, `flip`, `roll`                                                                                  |
| Cumulative  | `cumsum`, `cumprod`                                                                                                        |
| Permutation | `permute`, `shuffle`                                                                                                       |

## Slicing Operations

### Slicing Along Batch Dimension

Extract a subset of the batch:

```pycon
>>> import torch
>>> from batchtensor.tensor import slice_along_batch
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> # Take first 2 items
>>> slice_along_batch(tensor, stop=2)
tensor([[1, 2],
        [3, 4]])
>>> # Take items 1 to 3
>>> slice_along_batch(tensor, start=1, stop=3)
tensor([[3, 4],
        [5, 6]])
>>> # Take every other item
>>> slice_along_batch(tensor, step=2)
tensor([[1, 2],
        [5, 6]])

```

### Slicing Along Sequence Dimension

For sequential data:

```pycon
>>> import torch
>>> from batchtensor.tensor import slice_along_seq
>>> tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
>>> # Take first 2 timesteps
>>> slice_along_seq(tensor, stop=2)
tensor([[1, 2],
        [5, 6]])
>>> # Take last 2 timesteps
>>> slice_along_seq(tensor, start=2)
tensor([[3, 4],
        [7, 8]])

```

### Selecting Single Items

Select a specific index:

```pycon
>>> import torch
>>> from batchtensor.tensor import select_along_batch, select_along_seq
>>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> # Select second batch item (removes batch dimension)
>>> select_along_batch(tensor, index=1)
tensor([4, 5, 6])
>>> # Select first sequence position (removes sequence dimension)
>>> select_along_seq(tensor, index=0)
tensor([1, 4, 7])

```

### Chunking

Split tensors into equal chunks:

```pycon
>>> import torch
>>> from batchtensor.tensor import chunk_along_batch, chunk_along_seq
>>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
>>> chunks = chunk_along_batch(tensor, chunks=3)
>>> len(chunks)
3
>>> chunks[0]
tensor([[0, 1],
        [2, 3]])
>>> chunks[1]
tensor([[4, 5],
        [6, 7]])
>>> chunks[2]
tensor([[8, 9]])

```

### Splitting

Split tensors by specific sizes:

```pycon
>>> import torch
>>> from batchtensor.tensor import split_along_batch
>>> tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
>>> # Split into chunks of size 2
>>> splits = split_along_batch(tensor, split_size_or_sections=2)
>>> len(splits)
3
>>> splits[0]
tensor([[0, 1],
        [2, 3]])
>>> splits[1]
tensor([[4, 5],
        [6, 7]])
>>> splits[2]
tensor([[8, 9]])
>>> # Split with specific sizes
>>> splits = split_along_batch(tensor, split_size_or_sections=[2, 1, 2])
>>> len(splits)
3
>>> splits[0]
tensor([[0, 1],
        [2, 3]])
>>> splits[1]
tensor([[4, 5]])
>>> splits[2]
tensor([[6, 7],
        [8, 9]])

```

## Indexing Operations

### Index Selection

Select specific indices from a tensor:

```pycon
>>> import torch
>>> from batchtensor.tensor import index_select_along_batch
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
>>> indices = torch.tensor([2, 0, 1])
>>> index_select_along_batch(tensor, indices)
tensor([[5, 6],
        [1, 2],
        [3, 4]])

```

For sequences:

```pycon
>>> import torch
>>> from batchtensor.tensor import index_select_along_seq
>>> tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
>>> indices = torch.tensor([3, 0, 2])
>>> index_select_along_seq(tensor, indices)
tensor([[4, 1, 3],
        [8, 5, 7]])

```

## Joining Operations

### Concatenation

Combine multiple tensors:

```pycon
>>> import torch
>>> from batchtensor.tensor import cat_along_batch
>>> tensor1 = torch.tensor([[1, 2]])
>>> tensor2 = torch.tensor([[3, 4]])
>>> tensor3 = torch.tensor([[5, 6]])
>>> cat_along_batch([tensor1, tensor2, tensor3])
tensor([[1, 2],
        [3, 4],
        [5, 6]])

```

Concatenate along sequence dimension:

```pycon
>>> import torch
>>> from batchtensor.tensor import cat_along_seq
>>> tensor1 = torch.tensor([[1], [2]])
>>> tensor2 = torch.tensor([[3], [4]])
>>> cat_along_seq([tensor1, tensor2])
tensor([[1, 3],
        [2, 4]])

```

### Repetition

Use `repeat_along_batch` or `repeat_along_seq` to repeat the data along a dimension.

Repeat sequences:

```pycon
>>> import torch
>>> from batchtensor.tensor import repeat_along_seq
>>> tensor = torch.tensor([[1, 2], [3, 4]])
>>> repeat_along_seq(tensor, repeats=3)
tensor([[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]])

```

## Padding Operations

### Padding a Batch

Pad along the batch or sequence dimension, at the end (`side="right"`, default) or at the start
(`side="left"`):

```pycon
>>> import torch
>>> from batchtensor.tensor import pad_along_seq
>>> pad_along_seq(torch.tensor([[1, 2], [3, 4]]), 1, side="left")
tensor([[0, 1, 2],
        [0, 3, 4]])

```

### Building a Batch from Sequences

`pad_sequences` pads tensors of different lengths into one `(batch, seq, ...)` tensor, and
`lengths_to_mask` builds the mask of the padded batch:

```pycon
>>> import torch
>>> from batchtensor.tensor import lengths_to_mask, pad_sequences
>>> sequences = [torch.tensor([1, 2, 3]), torch.tensor([4])]
>>> pad_sequences(sequences)
tensor([[1, 2, 3],
        [4, 0, 0]])
>>> lengths_to_mask(torch.tensor([len(s) for s in sequences]))
tensor([[ True,  True,  True],
        [ True, False, False]])

```

### Masked Reductions

`masked_sum_along_seq`, `masked_mean_along_seq`, `masked_amax_along_seq` and
`masked_amin_along_seq` reduce only the valid steps of a padded batch. The `mask` is a `bool`
tensor of shape `(batch, seq)` (what `lengths_to_mask` returns) and is broadcast over the
trailing dimensions:

```pycon
>>> import torch
>>> from batchtensor.tensor import lengths_to_mask, masked_mean_along_seq, masked_amax_along_seq
>>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 9.0, 9.0]])
>>> mask = lengths_to_mask(torch.tensor([3, 1]))
>>> masked_mean_along_seq(tensor, mask)
tensor([2., 4.])
>>> masked_amax_along_seq(tensor, mask)
tensor([3., 4.])

```

A row without valid step gives `0` for the sum, `nan` for the mean and the lowest (`amax`) or the
highest (`amin`) value of the dtype. The mean requires a floating point tensor.

## Reduction Operations

### Sum and Mean

Compute sum along dimensions:

```pycon
>>> import torch
>>> from batchtensor.tensor import sum_along_batch, mean_along_batch
>>> tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
>>> sum_along_batch(tensor)
tensor([ 9., 12.])
>>> mean_along_batch(tensor)
tensor([3., 4.])
>>> # Keep dimensions
>>> sum_along_batch(tensor, keepdim=True)
tensor([[ 9., 12.]])

```

Along sequence dimension:

```pycon
>>> import torch
>>> from batchtensor.tensor import sum_along_seq, mean_along_seq
>>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> sum_along_seq(tensor)
tensor([ 6., 15.])
>>> mean_along_seq(tensor)
tensor([2., 5.])

```

### Product

Compute product along dimensions:

```pycon
>>> import torch
>>> from batchtensor.tensor import prod_along_batch
>>> tensor = torch.tensor([[2, 3], [4, 5]])
>>> prod_along_batch(tensor)
tensor([ 8, 15])

```

### Min and Max

Find minimum and maximum values:

```pycon
>>> import torch
>>> from batchtensor.tensor import (
...     amax_along_batch,
...     amin_along_batch,
...     max_along_batch,
...     min_along_batch,
... )
>>> tensor = torch.tensor([[1, 5], [3, 2], [4, 6]])
>>> amax_along_batch(tensor)
tensor([4, 6])
>>> amin_along_batch(tensor)
tensor([1, 2])
>>> # max and min return both values and indices
>>> max_along_batch(tensor)
torch.return_types.max(
values=tensor([4, 6]),
indices=tensor([2, 2]))
>>> min_along_batch(tensor)
torch.return_types.min(
values=tensor([1, 2]),
indices=tensor([0, 1]))

```

### ArgMax and ArgMin

Find indices of extrema:

```pycon
>>> import torch
>>> from batchtensor.tensor import argmax_along_batch, argmin_along_batch
>>> tensor = torch.tensor([[1, 5], [3, 2], [4, 6]])
>>> argmax_along_batch(tensor)
tensor([2, 2])
>>> argmin_along_batch(tensor)
tensor([0, 1])

```

### Median

Compute median values:

```pycon
>>> import torch
>>> from batchtensor.tensor import median_along_batch
>>> tensor = torch.tensor([[1, 5], [3, 2], [4, 6]])
>>> median_along_batch(tensor)
torch.return_types.median(
values=tensor([3, 5]),
indices=tensor([1, 0]))

```

## Comparison and Sorting

### Sorting

Sort tensors:

```pycon
>>> import torch
>>> from batchtensor.tensor import sort_along_batch, argsort_along_batch
>>> tensor = torch.tensor([[3, 1], [1, 4], [2, 2]])
>>> sort_along_batch(tensor)
torch.return_types.sort(
values=tensor([[1, 1],
               [2, 2],
               [3, 4]]),
indices=tensor([[1, 0],
                [2, 2],
                [0, 1]]))
>>> # Get indices only
>>> argsort_along_batch(tensor)
tensor([[1, 0],
        [2, 2],
        [0, 1]])
>>> # Sort in descending order
>>> sort_along_batch(tensor, descending=True)
torch.return_types.sort(
values=tensor([[3, 4],
               [2, 2],
               [1, 1]]),
indices=tensor([[0, 1],
                [2, 2],
                [1, 0]]))

```

## Mathematical Operations

### Cumulative Operations

Compute cumulative sums and products:

```pycon
>>> import torch
>>> from batchtensor.tensor import cumsum_along_batch, cumprod_along_batch
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
>>> cumsum_along_batch(tensor)
tensor([[ 1,  2],
        [ 4,  6],
        [ 9, 12]])
>>> cumprod_along_batch(tensor)
tensor([[ 1,  2],
        [ 3,  8],
        [15, 48]])

```

Along sequence dimension:

```pycon
>>> import torch
>>> from batchtensor.tensor import cumsum_along_seq
>>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> cumsum_along_seq(tensor)
tensor([[ 1,  3,  6],
        [ 4,  9, 15]])

```

## Permutation Operations

### Shuffling

Randomly permute batch items:

```pycon
>>> import torch
>>> from batchtensor.tensor import shuffle_along_batch
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
>>> # Results will be random
>>> shuffled = shuffle_along_batch(tensor)

```

For reproducible shuffling, use a generator:

```pycon
>>> import torch
>>> from batchtensor.tensor import shuffle_along_batch
>>> from coola.random import make_torch_generator
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
>>> generator = make_torch_generator(42)
>>> shuffled = shuffle_along_batch(tensor, generator=generator)

```

### Permuting with Specific Order

Apply a specific permutation:

```pycon
>>> import torch
>>> from batchtensor.tensor import permute_along_batch
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
>>> permutation = torch.tensor([2, 0, 1])
>>> permute_along_batch(tensor, permutation)
tensor([[5, 6],
        [1, 2],
        [3, 4]])

```

## Additional Reductions and Reordering

### std

`std_along_batch` and `std_along_seq` return the standard deviation along the batch or
the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import std_along_batch
>>> tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
>>> std_along_batch(tensor)
tensor([4.0000, 4.1633, 1.5275, 2.6458])

```

### var

`var_along_batch` and `var_along_seq` return the variance along the batch or the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import var_along_batch
>>> tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
>>> var_along_batch(tensor)
tensor([16.0000, 17.3333,  2.3333,  7.0000])

```

### any

`any_along_batch` and `any_along_seq` return whether any element is true along the batch or
the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import any_along_batch
>>> tensor = torch.tensor(
...     [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
... )
>>> any_along_batch(tensor)
tensor([True, True, True, True])

```

### all

`all_along_batch` and `all_along_seq` return whether all the elements are true along the batch or
the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import all_along_batch
>>> tensor = torch.tensor(
...     [[True, False, True, True], [True, True, True, True], [False, False, True, False]]
... )
>>> all_along_batch(tensor)
tensor([False, False,  True, False])

```

### logsumexp

`logsumexp_along_batch` and `logsumexp_along_seq` return the log of the sum of exponentials along
the batch or the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import logsumexp_along_batch
>>> tensor = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 9.0, 5.0, 6.0], [8.0, 7.0, 2.0, 1.0]])
>>> logsumexp_along_batch(tensor)
tensor([8.0185, 9.1272, 5.1698, 6.0247])

```

### flip

`flip_along_batch` and `flip_along_seq` return the tensor with the order of the elements reversed
along the batch or the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import flip_along_batch
>>> tensor = torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])
>>> flip_along_batch(tensor)
tensor([[8, 7, 2, 1],
        [4, 9, 5, 6],
        [0, 1, 3, 2]])

```

### roll

`roll_along_batch` and `roll_along_seq` return the tensor with the elements rolled along the batch
or the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import roll_along_batch
>>> tensor = torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])
>>> roll_along_batch(tensor, 1)
tensor([[8, 7, 2, 1],
        [0, 1, 3, 2],
        [4, 9, 5, 6]])

```

### topk

`topk_along_batch` and `topk_along_seq` return the k largest (or smallest) elements along the batch
or the sequence dimension.

```pycon
>>> import torch
>>> from batchtensor.tensor import topk_along_batch
>>> tensor = torch.tensor([[0, 1, 3, 2], [4, 9, 5, 6], [8, 7, 2, 1]])
>>> topk_along_batch(tensor, 2)
torch.return_types.topk(
values=tensor([[8, 9, 5, 6],
        [4, 7, 3, 2]]),
indices=tensor([[2, 1, 1, 1],
        [1, 2, 0, 0]]))

```

## Behaviour to know

### Views and copies

`slice`, `select`, `chunk` and `split` return views of the input, as torch does. Writing to the
output writes to the input. The other functions return new tensors, and no function modifies its
input. Even `pad_along_seq(x, 0)` returns a new tensor.

```pycon
>>> import torch
>>> from batchtensor.tensor import pad_along_seq, slice_along_batch
>>> x = torch.tensor([[1, 2], [3, 4]])
>>> view = slice_along_batch(x, stop=1)
>>> view[0, 0] = 9
>>> x
tensor([[9, 2],
        [3, 4]])
>>> pad_along_seq(x, 0) is x
False

```

### Options are keyword-only

The positional arguments are the data and the arguments that define the operation. Options such as
`keepdim`, `descending`, `stable`, `step`, `value`, `side`, `generator` and `max_len` must be passed
by keyword:

```pycon
>>> import torch
>>> from batchtensor.tensor import slice_along_batch, sort_along_seq
>>> x = torch.tensor([[3, 1, 2], [6, 5, 4]])
>>> sort_along_seq(x, descending=True).values
tensor([[3, 2, 1],
        [6, 5, 4]])
>>> slice_along_batch(torch.arange(6).reshape(6, 1), step=2)
tensor([[0],
        [2],
        [4]])

```

### Errors

An argument that torch cannot validate, or accepts with another meaning, is checked and raises a
`ValueError`. Everything else is the torch error, unchanged.

```pycon
>>> import torch
>>> from batchtensor.tensor import pad_along_seq, permute_along_batch
>>> permute_along_batch(torch.ones(3, 2), torch.tensor([0, 1]))
Traceback (most recent call last):
    ...
ValueError: ...
>>> pad_along_seq(torch.ones(2, 2, dtype=torch.long), 1, value=1.5)
Traceback (most recent call last):
    ...
ValueError: value (1.5) cannot be represented with dtype torch.int64

```

The full table is in [Design principles](../design_principles.md#one-error-contract-p8).

### Randomness

`shuffle_along_batch` and `shuffle_along_seq` draw a permutation. By default they use the global
torch random state. Pass `generator=` a `torch.Generator` or an `int` seed to make it reproducible:

```pycon
>>> import torch
>>> from batchtensor.tensor import shuffle_along_batch
>>> x = torch.arange(5).reshape(5, 1)
>>> bool((shuffle_along_batch(x, generator=42) == shuffle_along_batch(x, generator=42)).all())
True

```

### Devices and dtypes

No function moves data between devices or changes the dtype. The only function that can force a
host-device synchronization is `lengths_to_mask` without `max_len`, because it reads the largest
length. Pass `max_len` inside a training step on an accelerator.

### Edge cases

The results on degenerate input (a row of a masked reduction without valid step, a repeated index in
`permute_*`, `pad_size=0`, ...) are listed in
[Design principles](../design_principles.md#edge-cases-have-a-documented-result-p16).

## See Also

- [Nested Operations](nested.md) - Operations for nested data structures
- [Design principles](../design_principles.md) - Behaviour guarantees and trade-offs
- [API Reference](../refs/tensor.md) - Complete function reference
- [Constants](constants.md) - Dimension constants used throughout the library
