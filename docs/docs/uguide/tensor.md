# Tensor Operations

The `batchtensor.tensor` module provides functions to manipulate individual PyTorch tensors along
batch and sequence dimensions. These functions are the foundation for the nested operations and can
be used directly when working with single tensors.

## Overview

The tensor module operates on individual tensors with two primary dimensional conventions:

- **Batch dimension**: The first dimension (dimension 0) represents the batch
- **Sequence dimension**: The second dimension (dimension 1) represents the sequence/time steps

All functions in this module follow these conventions. The shape assumptions are:

- Batch operations: `(batch_size, *)` where `*` means any additional dimensions
- Sequence operations: `(batch_size, seq_len, *)` where `*` means any additional dimensions

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

Repeat sequences:

```pycon
>>> import torch
>>> from batchtensor.tensor import repeat_along_seq
>>> tensor = torch.tensor([[1, 2], [3, 4]])
>>> repeat_along_seq(tensor, repeats=3)
tensor([[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]])

```

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
>>> from batchtensor.utils.seed import get_torch_generator
>>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
>>> generator = get_torch_generator(42)
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

## Best Practices

1. **Dimension Awareness**: Always ensure your tensors follow the expected shape conventions
   (batch dimension first, sequence dimension second)
2. **Memory Efficiency**: Many operations like `slice` and `select` return views when possible,
   not copies
3. **Batch Processing**: These functions are optimized for batch processing, making them efficient
   for handling multiple samples simultaneously
4. **Consistency**: Use these functions instead of direct indexing to maintain code clarity and
   reduce errors

## Performance Considerations

- Operations leverage PyTorch's optimized tensor operations
- Views are used when possible to avoid data copying
- Batch operations are more efficient than processing items one at a time
- GPU acceleration is available through PyTorch's device management

## See Also

- [Nested Operations](nested.md) - Operations for nested data structures
- [API Reference](../refs/tensor.md) - Complete function reference
- [Constants](constants.md) - Dimension constants used throughout the library
