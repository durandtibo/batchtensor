# Constants

The `batchtensor.constants` module defines important constants used throughout the library to
identify batch and sequence dimensions.

## Overview

The constants module provides standardized dimension indices that ensure consistency across all
batchtensor operations. These constants are used internally by the library and can also be used in
your own code when working with batchtensor functions.

## Available Constants

### BATCH_DIM

The batch dimension constant identifies the dimension used for batching in tensors.

```pycon
>>> from batchtensor.constants import BATCH_DIM
>>> BATCH_DIM
0

```

**Usage:** This constant is set to `0`, indicating that the batch dimension is always the first
dimension (dimension 0) of tensors when using batchtensor functions.

**Convention:**

- For batch tensors: shape is `(batch_size, *)`
- The batch dimension contains independent samples
- Operations along this dimension process each sample in the batch

### SEQ_DIM

The sequence dimension constant identifies the dimension used for sequences in tensors.

```pycon
>>> from batchtensor.constants import SEQ_DIM
>>> SEQ_DIM
1

```

**Usage:** This constant is set to `1`, indicating that the sequence dimension is always the second
dimension (dimension 1) of tensors when using batchtensor functions.

**Convention:**

- For sequence tensors: shape is `(batch_size, seq_len, *)`
- The sequence dimension contains sequential/temporal data
- Operations along this dimension process time steps or sequence positions

## Why Use Constants?

Using constants instead of hard-coded numbers provides several benefits:

1. **Code Clarity**: `BATCH_DIM` is more readable than `0`
2. **Consistency**: Ensures all operations use the same dimension conventions
3. **Maintainability**: If dimension conventions ever change, updating the constant updates all uses
4. **Self-Documentation**: Code using these constants is self-explanatory

## Practical Examples

### Using Constants in Your Code

When working with batchtensor functions, you can use these constants to make your code more
explicit:

```pycon
>>> import torch
>>> from batchtensor.constants import BATCH_DIM, SEQ_DIM
>>> # Create a batch of sequences
>>> # Shape: (batch_size=2, seq_len=3, features=4)
>>> data = torch.randn(2, 3, 4)
>>> # Check dimensions
>>> batch_size = data.size(BATCH_DIM)
>>> seq_len = data.size(SEQ_DIM)
>>> print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
Batch size: 2, Sequence length: 3

```

### Manual Operations with Constants

When you need to perform operations directly with PyTorch but want to maintain consistency with
batchtensor conventions:

```pycon
>>> import torch
>>> from batchtensor.constants import BATCH_DIM, SEQ_DIM
>>> data = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
>>> # Sum along batch dimension
>>> batch_sum = data.sum(dim=BATCH_DIM)
>>> batch_sum
tensor([[ 6,  8],
        [10, 12]])
>>> # Sum along sequence dimension
>>> seq_sum = data.sum(dim=SEQ_DIM)
>>> seq_sum
tensor([[ 4,  6],
        [12, 14]])

```

### Verifying Tensor Shapes

Use constants to verify that your tensors have the expected shape:

```pycon
>>> import torch
>>> from batchtensor.constants import BATCH_DIM, SEQ_DIM
>>> def validate_batch_tensor(tensor, expected_batch_size):
...     """Validate that a tensor has the expected batch size."""
...     actual_batch_size = tensor.size(BATCH_DIM)
...     if actual_batch_size != expected_batch_size:
...         raise ValueError(
...             f"Expected batch size {expected_batch_size}, " f"got {actual_batch_size}"
...         )
...     return True
...
>>> tensor = torch.randn(32, 10)  # batch_size=32, features=10
>>> validate_batch_tensor(tensor, expected_batch_size=32)
True

```

### Creating Batches and Sequences

Use constants when manually creating or reshaping tensors:

```pycon
>>> import torch
>>> from batchtensor.constants import BATCH_DIM, SEQ_DIM
>>> # Create individual samples
>>> sample1 = torch.tensor([[1, 2], [3, 4]])  # seq_len=2, features=2
>>> sample2 = torch.tensor([[5, 6], [7, 8]])
>>> # Stack into a batch along batch dimension
>>> batch = torch.stack([sample1, sample2], dim=BATCH_DIM)
>>> batch.shape
torch.Size([2, 2, 2])
>>> # Verify dimensions
>>> print(f"Batch size: {batch.size(BATCH_DIM)}")
Batch size: 2
>>> print(f"Sequence length: {batch.size(SEQ_DIM)}")
Sequence length: 2

```

## Internal Usage

These constants are used internally throughout batchtensor. For example (simplified pseudocode):

```python
# In batchtensor.tensor.reduction
def sum_along_batch(tensor: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Sum tensor along the batch dimension."""
    return tensor.sum(dim=BATCH_DIM, keepdim=keepdim)


# In batchtensor.tensor.slicing (simplified for illustration)
def select_along_seq(tensor: torch.Tensor, index: int) -> torch.Tensor:
    """Select a specific index along the sequence dimension."""
    return tensor.select(dim=SEQ_DIM, index=index)
```

This ensures consistency across all functions in the library.

## Compatibility with Other Libraries

While batchtensor uses these specific dimension conventions, they are compatible with common PyTorch
practices:

- **Batch-first convention**: Most PyTorch modules (like `nn.Linear`, `nn.GRU` with
  `batch_first=True`) expect batch as the first dimension
- **Standard computer vision**: Images are typically `(batch, channels, height, width)`, compatible
  with `BATCH_DIM=0`
- **NLP sequence models**: With `batch_first=True`, sequences are `(batch, seq_len, features)`,
  matching our conventions

## Best Practices

1. **Import the constants** when you need to reference dimensions explicitly
2. **Use in assertions** to validate tensor shapes in your code
3. **Prefer batchtensor functions** over manual dimension handling when possible
4. **Document assumptions** about tensor shapes in your own functions

## See Also

- [Tensor Operations](tensor.md) - Functions that use these dimension constants
- [Nested Operations](nested.md) - Nested operations following the same conventions
