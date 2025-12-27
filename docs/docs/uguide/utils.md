# Utility Functions

The `batchtensor.utils` module provides utility functions for traversing and analyzing nested tensor structures. These functions are useful for debugging, introspection, and understanding the structure of complex nested data.

## Overview

The utils module includes functions for:

- Traversing nested structures using breadth-first search (BFS)
- Traversing nested structures using depth-first search (DFS)
- Analyzing the structure and contents of nested data

## Breadth-First Search (BFS)

The `bfs_tensor` function traverses a nested structure in breadth-first order, visiting all items at the current depth before moving to the next depth level.

### Basic Usage

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> data = {
...     "a": torch.tensor([1, 2, 3]),
...     "b": torch.tensor([4, 5, 6]),
... }
>>> for path, value in bfs_tensor(data):
...     print(f"{path}: {value.shape}")
...
('a',): torch.Size([3])
('b',): torch.Size([3])

```

### Nested Structures

BFS processes siblings before children:

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> data = {
...     "level1_a": torch.tensor([1, 2]),
...     "level1_b": {
...         "level2_a": torch.tensor([3, 4]),
...         "level2_b": torch.tensor([5, 6]),
...     },
... }
>>> for path, value in bfs_tensor(data):
...     print(f"Path: {path}, Shape: {value.shape}")
...
Path: ('level1_a',), Shape: torch.Size([2])
Path: ('level1_b', 'level2_a'), Shape: torch.Size([2])
Path: ('level1_b', 'level2_b'), Shape: torch.Size([2])

```

### Understanding the Traversal Order

BFS visits nodes level by level:

```
Structure:           Traversal Order:
    root                   1
   /  |  \                /  |  \
  A   B   C              2   3   4
     / \                    / \
    D   E                  5   6
```

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> data = {
...     "A": torch.tensor([1]),
...     "B": {
...         "D": torch.tensor([2]),
...         "E": torch.tensor([3]),
...     },
...     "C": torch.tensor([4]),
... }
>>> paths = [path for path, _ in bfs_tensor(data)]
>>> paths
[('A',), ('B', 'D'), ('B', 'E'), ('C',)]

```

## Depth-First Search (DFS)

The `dfs_tensor` function traverses a nested structure in depth-first order, exploring as far as possible along each branch before backtracking.

### Basic Usage

```pycon
>>> import torch
>>> from batchtensor.utils import dfs_tensor
>>> data = {
...     "a": torch.tensor([1, 2, 3]),
...     "b": torch.tensor([4, 5, 6]),
... }
>>> for path, value in dfs_tensor(data):
...     print(f"{path}: {value.shape}")
...
('a',): torch.Size([3])
('b',): torch.Size([3])

```

### Nested Structures

DFS explores deeply before moving to siblings:

```pycon
>>> import torch
>>> from batchtensor.utils import dfs_tensor
>>> data = {
...     "level1_a": torch.tensor([1, 2]),
...     "level1_b": {
...         "level2_a": torch.tensor([3, 4]),
...         "level2_b": torch.tensor([5, 6]),
...     },
... }
>>> for path, value in dfs_tensor(data):
...     print(f"Path: {path}, Shape: {value.shape}")
...
Path: ('level1_a',), Shape: torch.Size([2])
Path: ('level1_b', 'level2_a'), Shape: torch.Size([2])
Path: ('level1_b', 'level2_b'), Shape: torch.Size([2])

```

### Understanding the Traversal Order

DFS explores depth before breadth:

```
Structure:           Traversal Order:
    root                   1
   /  |  \                /  |  \
  A   B   C              2   4   6
     / \                    / \
    D   E                  3   5
```

```pycon
>>> import torch
>>> from batchtensor.utils import dfs_tensor
>>> data = {
...     "A": torch.tensor([1]),
...     "B": {
...         "D": torch.tensor([2]),
...         "E": torch.tensor([3]),
...     },
...     "C": torch.tensor([4]),
... }
>>> paths = [path for path, _ in dfs_tensor(data)]
>>> paths
[('A',), ('B', 'D'), ('B', 'E'), ('C',)]

```

## Practical Applications

### Inspecting Batch Structure

View all tensors in a batch:

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> batch = {
...     "inputs": {
...         "tokens": torch.randn(32, 128),
...         "attention_mask": torch.ones(32, 128),
...     },
...     "targets": torch.randint(0, 1000, (32,)),
...     "metadata": {
...         "ids": torch.arange(32),
...     },
... }
>>> print("Batch structure:")
Batch structure:
>>> for path, tensor in bfs_tensor(batch):
...     print(f"  {'.'.join(path)}: shape={tensor.shape}, dtype={tensor.dtype}")
...
  inputs.tokens: shape=torch.Size([32, 128]), dtype=torch.float32
  inputs.attention_mask: shape=torch.Size([32, 128]), dtype=torch.float32
  targets: shape=torch.Size([32]), dtype=torch.int64
  metadata.ids: shape=torch.Size([32]), dtype=torch.int64

```

### Collecting Statistics

Gather information about all tensors:

```pycon
>>> import torch
>>> from batchtensor.utils import dfs_tensor
>>> data = {
...     "features": torch.randn(100, 10),
...     "labels": torch.randint(0, 2, (100,)),
...     "weights": torch.rand(100),
... }
>>> total_params = 0
>>> for path, tensor in dfs_tensor(data):
...     num_elements = tensor.numel()
...     total_params += num_elements
...     print(f"{'.'.join(path)}: {num_elements} elements")
...
features: 1000 elements
labels: 100 elements
weights: 100 elements
>>> print(f"Total elements: {total_params}")
Total elements: 1200

```

### Finding Specific Tensors

Search for tensors matching criteria:

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> batch = {
...     "image": torch.randn(3, 224, 224),
...     "mask": torch.randint(0, 2, (224, 224)),
...     "label": torch.tensor(5),
... }
>>> # Find 2D or higher tensors
>>> print("Multi-dimensional tensors:")
Multi-dimensional tensors:
>>> for path, tensor in bfs_tensor(batch):
...     if tensor.ndim >= 2:
...         print(f"  {'.'.join(path)}: {tensor.shape}")
...
  image: torch.Size([3, 224, 224])
  mask: torch.Size([224, 224])

```

### Validating Batch Dimensions

Check that all tensors have consistent batch size:

```pycon
>>> import torch
>>> from batchtensor.utils import dfs_tensor
>>> def validate_batch_size(data, expected_batch_size):
...     for path, tensor in dfs_tensor(data):
...         if tensor.shape[0] != expected_batch_size:
...             return False, f"{'.'.join(path)} has batch size {tensor.shape[0]}"
...     return True, "All tensors have correct batch size"
...
>>> good_batch = {
...     "a": torch.randn(32, 10),
...     "b": torch.randn(32, 5),
... }
>>> validate_batch_size(good_batch, 32)
(True, 'All tensors have correct batch size')
>>> bad_batch = {
...     "a": torch.randn(32, 10),
...     "b": torch.randn(16, 5),  # Wrong batch size
... }
>>> validate_batch_size(bad_batch, 32)
(False, 'b has batch size 16')

```

### Memory Usage Analysis

Calculate memory footprint:

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> def calculate_memory(data):
...     total_bytes = 0
...     for path, tensor in bfs_tensor(data):
...         bytes_per_element = tensor.element_size()
...         num_elements = tensor.numel()
...         tensor_bytes = bytes_per_element * num_elements
...         total_bytes += tensor_bytes
...         print(f"{'.'.join(path)}: {tensor_bytes / 1024:.2f} KB")
...     return total_bytes / (1024 * 1024)
...
>>> batch = {
...     "features": torch.randn(1000, 512),  # float32
...     "labels": torch.randint(0, 10, (1000,)),  # int64
... }
>>> total_mb = calculate_memory(batch)
features: 2000.00 KB
labels: 7.81 KB
>>> print(f"Total: {total_mb:.2f} MB")
Total: 1.96 MB

```

## Comparing BFS vs DFS

### When to Use BFS

- When you want to process all items at the same nesting level together
- For level-order processing
- When shallow nodes are more important than deep ones
- Memory-efficient for wide but shallow structures

### When to Use DFS

- When you want to fully explore each branch before moving to the next
- For recursive-style processing
- When deep nodes are more important
- Memory-efficient for deep but narrow structures

### Example Comparison

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor, dfs_tensor
>>> data = {
...     "A": torch.tensor([1]),
...     "B": {
...         "C": {
...             "D": torch.tensor([2]),
...         },
...         "E": torch.tensor([3]),
...     },
... }
>>> print("BFS order:")
BFS order:
>>> for path, _ in bfs_tensor(data):
...     print(f"  {'.'.join(path)}")
...
  A
  B.C.D
  B.E
>>> print("DFS order:")
DFS order:
>>> for path, _ in dfs_tensor(data):
...     print(f"  {'.'.join(path)}")
...
  A
  B.C.D
  B.E

```

## Working with Lists and Tuples

Both functions handle list and tuple structures:

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> data = [
...     torch.tensor([1, 2]),
...     [
...         torch.tensor([3, 4]),
...         torch.tensor([5, 6]),
...     ],
... ]
>>> for path, tensor in bfs_tensor(data):
...     print(f"Index path {path}: {tensor.tolist()}")
...
Index path (0,): [1, 2]
Index path (1, 0): [3, 4]
Index path (1, 1): [5, 6]

```

## Best Practices

1. **Choose the Right Traversal**: Use BFS for level-order needs, DFS for branch-complete processing
2. **Path Tuples**: Paths are returned as tuples that can be used to access nested elements
3. **Generator Efficiency**: Both functions are generators, so they're memory efficient
4. **Type Checking**: Functions work with any nested structure, not just dicts
5. **Immutable Paths**: Don't modify the data structure during traversal

## Performance Considerations

- Both functions are generators and don't load all data into memory
- DFS typically uses less memory for deep structures
- BFS is better for structures with many siblings at each level
- For very large nested structures, consider iterating in chunks

## Common Patterns

### Print Structure

```pycon
>>> import torch
>>> from batchtensor.utils import bfs_tensor
>>> def print_structure(data, max_depth=None):
...     for path, tensor in bfs_tensor(data):
...         if max_depth is None or len(path) <= max_depth:
...             indent = "  " * (len(path) - 1)
...             name = path[-1]
...             print(f"{indent}{name}: {tensor.shape}")
...
>>> batch = {
...     "inputs": {"x": torch.randn(32, 10)},
...     "targets": torch.randint(0, 10, (32,)),
... }
>>> print_structure(batch)
x: torch.Size([32, 10])
targets: torch.Size([32])

```

### Extract All Tensors

```pycon
>>> import torch
>>> from batchtensor.utils import dfs_tensor
>>> data = {
...     "a": torch.tensor([1, 2]),
...     "b": {"c": torch.tensor([3, 4])},
... }
>>> tensors = [tensor for _, tensor in dfs_tensor(data)]
>>> len(tensors)
2

```

## See Also

- [Nested Operations](nested.md) - Operations on nested structures
- [Recursive Module](recursive.md) - Low-level recursive utilities
- [API Reference](../refs/utils.md) - Complete function reference
