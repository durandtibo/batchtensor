# Constants

The `batchtensor.constants` module defines important constants used throughout the library to
identify batch and sequence dimensions.

## Overview

The constants module provides the two dimension indices that every batchtensor function uses. The
library does not take a `dim` argument anywhere: the batch is always `BATCH_DIM` and the sequence is
always `SEQ_DIM`. You can use the constants in your own code to stay consistent with it.

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

## Why Constants?

The layout is a single definition that every function reads. Code that needs a dimension of a
batchtensor-style tensor, for example to call `torch` directly, can refer to the same definition
instead of a bare `0` or `1`, which makes the intent explicit. The values are fixed: they are not a
setting, and changing them is not supported.

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
...         raise ValueError(f"Expected batch size {expected_batch_size}, got {actual_batch_size}")
...     return True
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

The functions of `batchtensor.tensor` read the constants. For example (simplified):

```python
# In batchtensor.tensor.reduction
def sum_along_batch(tensor: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
    return tensor.sum(dim=BATCH_DIM, keepdim=keepdim)


# In batchtensor.tensor.slicing
def select_along_seq(tensor: torch.Tensor, index: int) -> torch.Tensor:
    return tensor.select(dim=SEQ_DIM, index=index)
```

## Compatibility with Other Libraries

While batchtensor uses these specific dimension conventions, they are compatible with common PyTorch
practices:

- **Batch-first convention**: Most PyTorch modules (like `nn.Linear`, `nn.GRU` with
  `batch_first=True`) expect batch as the first dimension
- **Standard computer vision**: Images are typically `(batch, channels, height, width)`, compatible
  with `BATCH_DIM=0`
- **NLP sequence models**: With `batch_first=True`, sequences are `(batch, seq_len, features)`,
  matching our conventions

## Data in Another Layout

Data laid out as `(seq, batch, ...)`, such as the default input of `torch.nn.LSTM`, must be
transposed before it is passed to batchtensor. For any operation on another dimension, call `torch`
directly. See [Design principles](../design_principles.md#the-layout-is-fixed-p1) for the trade-off.

```pycon
>>> import torch
>>> from batchtensor.tensor import sum_along_seq
>>> seq_first = torch.ones(5, 2, 3)  # (seq, batch, features)
>>> sum_along_seq(seq_first.transpose(0, 1))  # (batch, seq, features) -> (batch, features)
tensor([[5., 5., 5.],
        [5., 5., 5.]])

```

## See Also

- [Tensor Operations](tensor.md) - Functions that use these dimension constants
- [Nested Operations](nested.md) - Nested operations following the same conventions
- [Design principles](../design_principles.md) - Why the layout is fixed
