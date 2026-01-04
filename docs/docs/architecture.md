# Architecture and Design

This document provides an overview of the `batchtensor` library architecture, design principles, and
implementation details.

## Library Structure

The `batchtensor` library is organized into four main modules:

```
batchtensor/
├── nested/         # Operations for nested data structures
├── tensor/         # Operations for individual tensors
├── utils/          # Utility functions (seed management)
└── constants.py    # Dimension constants
```

## Design Principles

### 1. Consistency

All functions in batchtensor follow consistent conventions:

- **Batch dimension is always dimension 0**: The first dimension of tensors represents the batch
- **Sequence dimension is always dimension 1**: The second dimension represents sequences/time steps
- **Function naming**: Functions are named with the pattern `operation_along_dimension`

### 2. Separation of Concerns

The library separates operations into two levels:

- **`tensor` module**: Low-level operations on individual tensors
- **`nested` module**: High-level operations that recursively apply tensor operations to nested
  structures

This separation allows users to:

- Use tensor operations directly when working with single tensors
- Use nested operations when working with complex data structures
- Compose operations from both modules as needed

### 3. Minimal Dependencies

The library has minimal dependencies:

- **PyTorch**: Core tensor operations
- **coola**: Recursive operations on nested structures

This keeps the library lightweight and reduces potential conflicts with other packages.

### 4. Type Safety

All functions include comprehensive type hints:

- Input and output types are explicitly declared
- Generic types are used appropriately
- TYPE_CHECKING blocks avoid runtime overhead

## Module Details

### Constants Module

Defines dimension indices used throughout the library:

- `BATCH_DIM = 0`: Identifies the batch dimension
- `SEQ_DIM = 1`: Identifies the sequence dimension

Using constants instead of magic numbers improves code clarity and maintainability.

### Tensor Module

The tensor module provides low-level operations for individual tensors. It's organized into
sub-modules by operation type:

- **slicing.py**: Slice, chunk, split operations
- **indexing.py**: Index selection operations
- **joining.py**: Concatenation and repetition
- **reduction.py**: Sum, mean, min, max, median, etc.
- **comparison.py**: Sorting and comparison
- **math.py**: Cumulative operations
- **permutation.py**: Shuffling and permuting

Each function:

- Operates on a single PyTorch tensor
- Assumes standard dimension conventions (batch=0, seq=1)
- Returns a new tensor (or tuple of tensors)
- Includes comprehensive docstrings with examples

### Nested Module

The nested module provides high-level operations for nested data structures. It's organized to
mirror the tensor module:

- **slicing.py**: Nested slicing operations
- **indexing.py**: Nested index selection
- **joining.py**: Nested concatenation and repetition
- **reduction.py**: Nested reductions
- **comparison.py**: Nested sorting
- **math.py**: Nested cumulative operations
- **permutation.py**: Nested shuffling and permuting
- **conversion.py**: NumPy conversion
- **pointwise.py**: Element-wise operations
- **trigo.py**: Trigonometric functions
- **misc.py**: Miscellaneous utilities

Each nested function:

- Recursively applies the corresponding tensor operation
- Preserves the nested structure (dict, list, tuple)
- Uses `coola.recursive_apply` for recursive traversal
- Handles arbitrary nesting depth

### Utils Module

The utils module provides supporting functionality:

- **seed.py**: Random seed management for reproducibility
  - `get_random_seed()`: Generate deterministic random seeds
  - `get_torch_generator()`: Create PyTorch generators
  - `setup_torch_generator()`: Flexible generator setup

## Implementation Patterns

### Pattern 1: Tensor Operations Use Constants

```python
from batchtensor.constants import BATCH_DIM

def sum_along_batch(tensor, keepdim=False):
    return tensor.sum(dim=BATCH_DIM, keepdim=keepdim)
```

This ensures consistency and makes the code self-documenting.

### Pattern 2: Nested Operations Delegate to Tensor Operations

```python
from coola.recursive import recursive_apply
from batchtensor import tensor as bt

def slice_along_batch(data, start=None, stop=None, step=None):
    return recursive_apply(
        data,
        partial(bt.slice_along_batch, start=start, stop=stop, step=step)
    )
```

This reduces code duplication and ensures nested operations behave consistently.

### Pattern 3: Dictionary Operations Preserve Structure

```python
def chunk_along_batch(data, chunks):
    keys = data.keys()
    return tuple(
        dict(zip(keys, values))
        for values in zip(*[bt.chunk_along_batch(tensor, chunks) for tensor in data.values()])
    )
```

This pattern ensures the output structure matches the input structure.

## Extension Points

The library can be extended in several ways:

### Adding New Tensor Operations

1. Add the function to the appropriate sub-module in `tensor/`
2. Follow existing naming conventions
3. Include comprehensive docstring with example
4. Export from `tensor/__init__.py`

### Adding New Nested Operations

1. Add the corresponding function to `nested/`
2. Use `recursive_apply` to delegate to tensor operations
3. Export from `nested/__init__.py`

### Adding New Data Types

The nested operations work with any data structure that `coola.recursive_apply` supports:

- Dictionaries
- Lists
- Tuples
- Custom classes (with appropriate handlers)

## Performance Considerations

### Memory Efficiency

- Operations use views when possible (e.g., `slice`, `select`)
- Avoid unnecessary copies
- Leverage PyTorch's memory management

### Computational Efficiency

- Delegate to PyTorch's optimized operations
- Minimize Python overhead
- Support GPU acceleration through PyTorch

### Nested Structure Overhead

- Recursive operations have minimal overhead
- Dictionary access is O(1)
- Most time is spent in PyTorch operations, not structure traversal

## Testing Strategy

The library uses comprehensive testing:

- **Unit tests**: Test individual functions with various inputs
- **Integration tests**: Test combinations of operations
- **Doctests**: Verify examples in docstrings
- **Type checking**: Use pyright for static type checking

## Future Directions

Potential areas for expansion:

1. **Additional operations**: More mathematical and statistical functions
2. **Custom data structures**: Support for more complex nested types
3. **Performance optimizations**: Specialized implementations for common patterns
4. **Batch dataset utilities**: Higher-level abstractions for common workflows

## See Also

- [Tensor Operations Guide](uguide/tensor.md)
- [Nested Operations Guide](uguide/nested.md)
- [Utils Guide](uguide/utils.md)
- [Constants Documentation](uguide/constants.md)
