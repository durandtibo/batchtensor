# Recursive Operations

The `batchtensor.recursive` module provides low-level utilities for applying functions recursively to nested data structures. This module is used internally by higher-level functions but can also be used directly for custom operations.

## Overview

The recursive module enables you to apply any function to elements within nested structures like dictionaries, lists, and tuples. It uses a pluggable applier system that handles different data types appropriately.

## Basic Usage

### The `recursive_apply` Function

The core function for recursive operations:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> def double(x):
...     return x * 2
...
>>> data = {
...     "a": torch.tensor([1, 2, 3]),
...     "b": torch.tensor([4, 5, 6]),
... }
>>> recursive_apply(data, double)
{'a': tensor([2, 4, 6]), 'b': tensor([ 8, 10, 12])}

```

### Nested Structures

Works with deeply nested data:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> data = {
...     "level1": {
...         "level2": torch.tensor([1, 2]),
...         "level2b": torch.tensor([3, 4]),
...     },
...     "other": torch.tensor([5, 6]),
... }
>>> recursive_apply(data, lambda x: x + 10)
{'level1': {'level2': tensor([11, 12]), 'level2b': tensor([13, 14])}, 'other': tensor([15, 16])}

```

### Lists and Tuples

Handles sequences naturally:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> data = [
...     torch.tensor([1, 2]),
...     torch.tensor([3, 4]),
... ]
>>> recursive_apply(data, lambda x: x**2)
[tensor([1, 4]), tensor([9, 16])]

```

## Applier System

The recursive module uses an applier pattern to handle different data types. Each applier knows how to traverse a specific type of data structure.

### AutoApplier

The `AutoApplier` automatically selects the correct applier based on data type:

```pycon
>>> from batchtensor.recursive import AutoApplier
>>> # Check if an applier is registered for a type
>>> AutoApplier.has_applier(dict)
True
>>> AutoApplier.has_applier(list)
True
>>> AutoApplier.has_applier(str)
False

```

### Built-in Appliers

Several appliers are provided:

- **`DefaultApplier`**: Applies function directly to the data (leaf nodes)
- **`MappingApplier`**: Handles dict-like objects
- **`SequenceApplier`**: Handles list and tuple objects
- **`AutoApplier`**: Automatically chooses the appropriate applier

```pycon
>>> from batchtensor.recursive import (
...     DefaultApplier,
...     MappingApplier,
...     SequenceApplier,
... )
>>> # View an applier
>>> MappingApplier()
MappingApplier()
>>> SequenceApplier()
SequenceApplier()

```

### Custom Appliers

You can create custom appliers for your own types by inheriting from `BaseApplier`:

```pycon
>>> from batchtensor.recursive import BaseApplier, AutoApplier, ApplyState
>>> import torch
>>> class MyClass:
...     def __init__(self, value):
...         self.value = value
...
>>> class MyClassApplier(BaseApplier):
...     def apply(self, data, func, state):
...         # Apply function to the value attribute
...         data.value = func(data.value)
...         return data
...
>>> # Register the custom applier
>>> AutoApplier.add_applier(MyClass, MyClassApplier(), exist_ok=True)
>>> # Now use it
>>> obj = MyClass(torch.tensor([1, 2, 3]))
>>> from batchtensor.recursive import recursive_apply
>>> result = recursive_apply(obj, lambda x: x * 2)
>>> result.value
tensor([2, 4, 6])

```

## ApplyState

The `ApplyState` class tracks the recursion state during application:

```pycon
>>> from batchtensor.recursive import ApplyState
>>> state = ApplyState()
>>> state
ApplyState()

```

This is mainly used internally to manage recursion context, but you can access it if needed for advanced use cases.

## Practical Examples

### Apply Function to Specific Keys

Filter which tensors get modified:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> def scale_features(data):
...     if isinstance(data, dict):
...         result = {}
...         for key, value in data.items():
...             if key.startswith("feature"):
...                 result[key] = recursive_apply(value, lambda x: x * 0.1)
...             else:
...                 result[key] = value
...         return result
...     return data
...
>>> batch = {
...     "feature1": torch.tensor([10.0, 20.0]),
...     "feature2": torch.tensor([30.0, 40.0]),
...     "label": torch.tensor([0, 1]),
... }
>>> scale_features(batch)
{'feature1': tensor([1., 2.]), 'feature2': tensor([3., 4.]), 'label': tensor([0, 1])}

```

### Normalize Nested Data

Apply normalization recursively:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> def normalize(x):
...     if isinstance(x, torch.Tensor) and x.dtype in [torch.float32, torch.float64]:
...         return (x - x.mean()) / (x.std() + 1e-8)
...     return x
...
>>> data = {
...     "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
...     "metadata": {
...         "counts": torch.tensor([10, 20, 30], dtype=torch.int64),
...         "scores": torch.tensor([0.5, 1.0, 1.5]),
...     },
... }
>>> recursive_apply(data, normalize)  # doctest: +SKIP
{'input': tensor([-1.4142, -0.7071,  0.0000,  0.7071,  1.4142]), 'metadata': {'counts': tensor([10, 20, 30]), 'scores': tensor([-1.2247,  0.0000,  1.2247])}}

```

### Collect Statistics

Gather information from nested structures:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> shapes = []
>>> def collect_shapes(x):
...     if isinstance(x, torch.Tensor):
...         shapes.append(x.shape)
...     return x
...
>>> data = {
...     "a": torch.tensor([[1, 2], [3, 4]]),
...     "b": torch.tensor([5, 6, 7]),
... }
>>> _ = recursive_apply(data, collect_shapes)
>>> shapes
[torch.Size([2, 2]), torch.Size([3])]

```

## Advanced Usage

### Combining with Higher-Level Functions

The recursive module is used internally by nested operations:

```pycon
>>> # This is how nested functions work internally
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> from functools import partial
>>> # The nested module uses recursive_apply
>>> def slice_all(data, start, stop):
...     return recursive_apply(data, partial(lambda x, s, e: x[s:e], s=start, e=stop))
...
>>> batch = {
...     "a": torch.tensor([[1, 2], [3, 4], [5, 6]]),
...     "b": torch.tensor([7, 8, 9]),
... }
>>> slice_all(batch, 0, 2)
{'a': tensor([[1, 2], [3, 4]]), 'b': tensor([7, 8])}

```

### Type-Safe Operations

Ensure operations only apply to tensors:

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> def safe_operation(x):
...     if isinstance(x, torch.Tensor):
...         return x.float()
...     return x
...
>>> data = {
...     "tensor": torch.tensor([1, 2, 3], dtype=torch.int32),
...     "string": "metadata",
...     "number": 42,
... }
>>> recursive_apply(data, safe_operation)
{'tensor': tensor([1., 2., 3.]), 'string': 'metadata', 'number': 42}

```

## Best Practices

1. **Keep Functions Pure**: Functions passed to `recursive_apply` should not have side effects on the structure itself
2. **Handle Multiple Types**: If your function might encounter different types, add type checks
3. **Use Partial Functions**: When you need to pass additional arguments, use `functools.partial`
4. **Register Custom Appliers Early**: Add custom appliers at module initialization
5. **Avoid Deep Recursion**: Very deeply nested structures may hit recursion limits

## Performance Considerations

- The applier pattern adds minimal overhead
- Most time is spent in your custom function
- For large batches, vectorized operations are much faster than recursive calls
- Consider flattening data if recursion depth is very large

## Common Patterns

### Transform and Filter

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> # Only modify float tensors
>>> def to_half_precision(x):
...     if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
...         return x.half()
...     return x
...
>>> data = {
...     "floats": torch.tensor([1.0, 2.0], dtype=torch.float32),
...     "ints": torch.tensor([1, 2], dtype=torch.int32),
... }
>>> recursive_apply(data, to_half_precision)
{'floats': tensor([1., 2.], dtype=torch.float16), 'ints': tensor([1, 2], dtype=torch.int32)}

```

### Clone Nested Data

```pycon
>>> import torch
>>> from batchtensor.recursive import recursive_apply
>>> data = {
...     "a": torch.tensor([1, 2, 3]),
...     "b": {"c": torch.tensor([4, 5])},
... }
>>> cloned = recursive_apply(
...     data, lambda x: x.clone() if isinstance(x, torch.Tensor) else x
... )
>>> cloned["a"][0] = 100
>>> data["a"][0]  # Original unchanged
tensor(1)

```

## See Also

- [Nested Operations](nested.md) - High-level operations using recursive utilities
- [Utils Module](utils.md) - Tensor traversal utilities
- [API Reference](../refs/recursive.md) - Complete function reference
