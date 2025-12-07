# Nested Data Manipulation

The `batchtensor.nested` module provides functions to manipulate nested data structures containing PyTorch tensors. These functions work with dictionaries, lists, and other nested structures where tensors share batch or sequence dimensions.

## Overview

When working with complex data pipelines, you often have batches represented as nested structures (e.g., dictionaries of tensors). The nested module makes it easy to apply operations across all tensors in these structures.

## Slicing Operations

### Slicing Along Batch Dimension

Extract a subset of the batch from all tensors:

```pycon
>>> import torch
>>> from batchtensor.nested import slice_along_batch
>>> batch = {
...     "features": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
...     "labels": torch.tensor([0, 1, 0, 1]),
...     "weights": torch.tensor([1.0, 0.5, 0.8, 1.2]),
... }
>>> # Take first 2 items
>>> slice_along_batch(batch, stop=2)
{'features': tensor([[1, 2], [3, 4]]), 'labels': tensor([0, 1]), 'weights': tensor([1.0000, 0.5000])}

```

### Slicing Along Sequence Dimension

For sequential data with shape `(batch_size, seq_len, *)`:

```pycon
>>> import torch
>>> from batchtensor.nested import slice_along_seq
>>> batch = {
...     "tokens": torch.tensor([[[1], [2], [3], [4]], [[5], [6], [7], [8]]]),
...     "attention": torch.tensor([[1.0, 0.9, 0.8, 0.7], [1.0, 0.95, 0.9, 0.85]]),
... }
>>> # Take first 2 timesteps
>>> slice_along_seq(batch, stop=2)
{'tokens': tensor([[[1], [2]], [[5], [6]]]), 'attention': tensor([[1.0000, 0.9000], [1.0000, 0.9500]])}

```

### Chunking

Split tensors into equal chunks:

```pycon
>>> import torch
>>> from batchtensor.nested import chunk_along_batch
>>> batch = {
...     "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
...     "b": torch.tensor([4, 3, 2, 1, 0]),
... }
>>> chunks = chunk_along_batch(batch, chunks=3)
>>> len(chunks)
3
>>> chunks[0]
{'a': tensor([[0, 1], [2, 3]]), 'b': tensor([4, 3])}

```

### Splitting

Split tensors by size:

```pycon
>>> import torch
>>> from batchtensor.nested import split_along_batch
>>> batch = {
...     "a": torch.tensor([[2, 6], [0, 3], [4, 9], [8, 1], [5, 7]]),
...     "b": torch.tensor([4, 3, 2, 1, 0]),
... }
>>> splits = split_along_batch(batch, split_size_or_sections=2)
>>> len(splits)
3
>>> splits[0]
{'a': tensor([[2, 6], [0, 3]]), 'b': tensor([4, 3])}

```

## Indexing Operations

Select specific indices from all tensors:

```pycon
>>> import torch
>>> from batchtensor.nested import index_select_along_batch
>>> batch = {
...     "features": torch.tensor([[1, 2], [3, 4], [5, 6]]),
...     "labels": torch.tensor([0, 1, 2]),
... }
>>> indices = torch.tensor([2, 0])
>>> index_select_along_batch(batch, indices)
{'features': tensor([[5, 6], [1, 2]]), 'labels': tensor([2, 0])}

```

## Joining Operations

### Concatenation

Combine multiple batches:

```pycon
>>> import torch
>>> from batchtensor.nested import cat_along_batch
>>> batch1 = {"a": torch.tensor([[1, 2]]), "b": torch.tensor([10])}
>>> batch2 = {"a": torch.tensor([[3, 4]]), "b": torch.tensor([20])}
>>> cat_along_batch([batch1, batch2])
{'a': tensor([[1, 2], [3, 4]]), 'b': tensor([10, 20])}

```

### Repetition

Repeat sequences:

```pycon
>>> import torch
>>> from batchtensor.nested import repeat_along_seq
>>> batch = {
...     "seq": torch.tensor([[[1], [2]], [[3], [4]]]),
... }
>>> repeat_along_seq(batch, 2)
{'seq': tensor([[[1], [2], [1], [2]], [[3], [4], [3], [4]]])}

```

## Reduction Operations

### Sum, Mean, Min, Max

Compute statistics along dimensions:

```pycon
>>> import torch
>>> from batchtensor.nested import (
...     sum_along_batch,
...     mean_along_batch,
...     amax_along_batch,
...     amin_along_batch,
... )
>>> batch = {
...     "scores": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
...     "counts": torch.tensor([10, 20, 30]),
... }
>>> sum_along_batch(batch)
{'scores': tensor([9., 12.]), 'counts': tensor(60)}
>>> mean_along_batch(batch)
{'scores': tensor([3., 4.]), 'counts': tensor(20.)}

```

### ArgMax and ArgMin

Find indices of extrema:

```pycon
>>> import torch
>>> from batchtensor.nested import argmax_along_batch, argmin_along_batch
>>> batch = {"values": torch.tensor([[1, 5], [3, 2], [4, 6]])}
>>> argmax_along_batch(batch)
{'values': tensor([2, 2])}
>>> argmin_along_batch(batch)
{'values': tensor([0, 1])}

```

## Comparison and Sorting

Sort tensors:

```pycon
>>> import torch
>>> from batchtensor.nested import sort_along_batch
>>> batch = {
...     "values": torch.tensor([[3, 1], [1, 4], [2, 2]]),
... }
>>> values, indices = sort_along_batch(batch)
>>> values
{'values': tensor([[1, 1], [2, 2], [3, 4]])}
>>> indices
{'values': tensor([[1, 0], [2, 2], [0, 1]])}

```

## Mathematical Operations

### Cumulative Operations

```pycon
>>> import torch
>>> from batchtensor.nested import cumsum_along_batch, cumprod_along_batch
>>> batch = {"values": torch.tensor([[1, 2], [3, 4], [5, 6]])}
>>> cumsum_along_batch(batch)
{'values': tensor([[1, 2], [4, 6], [9, 12]])}

```

### Trigonometric Functions

Apply trigonometric functions element-wise:

```pycon
>>> import torch
>>> from batchtensor.nested import sin, cos, tan
>>> batch = {"angles": torch.tensor([[0.0, 1.57], [3.14, 4.71]])}
>>> sin(batch)
{'angles': tensor([[0.0000, 1.0000], [0.0016, -1.0000]])}

```

## Pointwise Operations

Apply element-wise operations:

```pycon
>>> import torch
>>> from batchtensor.nested import abs, exp, log, sqrt
>>> batch = {"values": torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])}
>>> abs(batch)
{'values': tensor([[1., 2.], [3., 4.]])}

```

## Permutation Operations

### Shuffling

Randomly permute batch items:

```pycon
>>> import torch
>>> from batchtensor.nested import shuffle_along_batch
>>> batch = {
...     "features": torch.tensor([[1, 2], [3, 4], [5, 6]]),
...     "labels": torch.tensor([0, 1, 2]),
... }
>>> # Results will be random
>>> shuffle_along_batch(batch)  # doctest: +SKIP
{'features': tensor([[5, 6], [1, 2], [3, 4]]), 'labels': tensor([2, 0, 1])}

```

### Permuting

Apply a specific permutation:

```pycon
>>> import torch
>>> from batchtensor.nested import permute_along_batch
>>> batch = {
...     "features": torch.tensor([[1, 2], [3, 4], [5, 6]]),
...     "labels": torch.tensor([0, 1, 2]),
... }
>>> permutation = torch.tensor([2, 0, 1])
>>> permute_along_batch(batch, permutation)
{'features': tensor([[5, 6], [1, 2], [3, 4]]), 'labels': tensor([2, 0, 1])}

```

## Type Conversion

### Change Device

Move all tensors to a specific device:

```pycon
>>> import torch
>>> from batchtensor.nested import to_device
>>> batch = {
...     "features": torch.tensor([[1, 2], [3, 4]]),
...     "labels": torch.tensor([0, 1]),
... }
>>> # Move to CPU (already there in this example)
>>> to_device(batch, torch.device("cpu"))
{'features': tensor([[1, 2], [3, 4]]), 'labels': tensor([0, 1])}

```

### Change Data Type

Convert tensor dtypes:

```pycon
>>> import torch
>>> from batchtensor.nested import to_dtype
>>> batch = {"values": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)}
>>> to_dtype(batch, dtype=torch.float32)
{'values': tensor([[1., 2.], [3., 4.]])}

```

## NumPy Conversion

Convert between PyTorch tensors and NumPy arrays:

```pycon
>>> import torch
>>> from batchtensor.nested import from_numpy, to_numpy
>>> import numpy as np
>>> # From NumPy
>>> numpy_data = {
...     "a": np.array([[1, 2], [3, 4]]),
...     "b": np.array([5, 6]),
... }
>>> batch = from_numpy(numpy_data)  # doctest: +SKIP
>>> batch  # doctest: +SKIP
{'a': tensor([[1, 2], [3, 4]]), 'b': tensor([5, 6])}
>>> # To NumPy
>>> to_numpy(batch)  # doctest: +SKIP
{'a': array([[1, 2], [3, 4]]), 'b': array([5, 6])}

```

## Best Practices

1. **Consistent Dimensions**: Ensure all tensors in your nested structure have compatible batch/sequence dimensions
2. **Memory Efficiency**: Many operations like `chunk` and `slice` return views when possible, not copies
3. **Type Safety**: Use the same data types across tensors in a batch for predictable behavior
4. **Error Handling**: Functions will raise clear errors if tensors have incompatible shapes

## Performance Considerations

- Operations are applied recursively to nested structures
- Views are used when possible to avoid copying data
- All functions leverage PyTorch's efficient tensor operations
- For very deep nesting, consider flattening your data structure

## See Also

- [Tensor Operations](tensor.md) - Operations for single tensors
- [Recursive Module](recursive.md) - Low-level recursive utilities
- [API Reference](../refs/nested.md) - Complete function reference
