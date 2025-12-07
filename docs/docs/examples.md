# Examples and Tutorials

This page provides practical examples of using `batchtensor` in real-world scenarios.

## Example 1: Processing Image Batches

Working with batches of images and their metadata:

```python
import torch
from batchtensor.nested import (
    slice_along_batch,
    shuffle_along_batch,
    split_along_batch,
)

# Create a batch of images with metadata
batch = {
    "images": torch.randn(100, 3, 224, 224),  # 100 RGB images
    "labels": torch.randint(0, 10, (100,)),    # Classification labels
    "filenames": ["img_{:03d}.jpg".format(i) for i in range(100)],
    "augmented": torch.randint(0, 2, (100,), dtype=torch.bool),
}

# Shuffle the entire batch while maintaining correspondence
shuffled = shuffle_along_batch(batch)

# Take the first 32 samples for a mini-batch
mini_batch = slice_along_batch(batch, stop=32)

# Split into train/val/test sets (60/20/20)
train, val, test = split_along_batch(batch, split_size_or_sections=[60, 20, 20])

print(f"Train set: {train['images'].shape[0]} images")
print(f"Val set: {val['images'].shape[0]} images")
print(f"Test set: {test['images'].shape[0]} images")
```

## Example 2: Sequence Data with Padding

Handling sequences of variable length:

```python
import torch
from batchtensor.nested import (
    cat_along_seq,
    slice_along_seq,
    select_along_seq,
)

# Batch of sequences with padding
batch = {
    "tokens": torch.tensor([
        [[1], [2], [3], [0]],  # sequence length 3, padded
        [[4], [5], [0], [0]],  # sequence length 2, padded
        [[6], [7], [8], [9]],  # sequence length 4, no padding
    ]),
    "lengths": torch.tensor([3, 2, 4]),  # actual lengths
}

# Take first 3 timesteps
truncated = slice_along_seq(batch, stop=3)
print(f"Truncated shape: {truncated['tokens'].shape}")

# Select specific timesteps
first_and_last = select_along_seq(batch, torch.tensor([0, 3]))
print(f"Selected shape: {first_and_last['tokens'].shape}")
```

## Example 3: Multi-Task Learning

Managing data for multiple tasks:

```python
import torch
from batchtensor.nested import (
    to_device,
    to_dtype,
    index_select_along_batch,
)

# Multi-task batch
batch = {
    "shared_features": torch.randn(64, 512),
    "task1": {
        "labels": torch.randint(0, 10, (64,)),
        "weights": torch.rand(64),
    },
    "task2": {
        "labels": torch.randint(0, 2, (64,)),
        "weights": torch.rand(64),
    },
}

# Move everything to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_gpu = to_device(batch, device)

# Convert all weights to float16 for mixed precision
batch_gpu["task1"] = to_dtype(batch_gpu["task1"], dtype=torch.float16)
batch_gpu["task2"] = to_dtype(batch_gpu["task2"], dtype=torch.float16)

# Sample a subset for task-specific training
task1_indices = torch.randperm(64)[:32]
task1_batch = index_select_along_batch(batch_gpu, task1_indices)
```

## Example 4: Data Augmentation Pipeline

Creating augmented views of data:

```python
import torch
from batchtensor.nested import cat_along_batch
from batchtensor.recursive import recursive_apply

def augment_image(image):
    """Simple augmentation: random crop and flip."""
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        image = torch.flip(image, dims=[-1])
    # Random crop (simplified)
    return image

# Original batch
original = {
    "images": torch.randn(32, 3, 224, 224),
    "labels": torch.randint(0, 1000, (32,)),
}

# Create augmented version
augmented = {
    "images": recursive_apply(
        original["images"],
        lambda x: augment_image(x) if isinstance(x, torch.Tensor) else x
    ),
    "labels": original["labels"],  # Labels stay the same
}

# Combine original and augmented
combined = cat_along_batch([original, augmented])
print(f"Combined batch size: {combined['images'].shape[0]}")
```

## Example 5: Temporal Data Processing

Working with time series data:

```python
import torch
from batchtensor.nested import (
    split_along_seq,
    chunk_along_seq,
    repeat_along_seq,
)

# Time series batch (batch_size=16, seq_len=100, features=10)
batch = {
    "sequences": torch.randn(16, 100, 10),
    "timestamps": torch.arange(100).unsqueeze(0).repeat(16, 1),
}

# Split into past and future
past, future = split_along_seq(batch, split_size_or_sections=[80, 20])
print(f"Past: {past['sequences'].shape}, Future: {future['sequences'].shape}")

# Create overlapping windows
windows = chunk_along_seq(batch, chunks=10)
print(f"Number of windows: {len(windows)}")

# Repeat sequences for ensemble predictions
ensemble_batch = repeat_along_seq(batch, 5)
print(f"Ensemble shape: {ensemble_batch['sequences'].shape}")
```

## Example 6: Batch Normalization

Computing statistics across the batch:

```python
import torch
from batchtensor.nested import (
    mean_along_batch,
    std_along_batch,
)
from batchtensor.recursive import recursive_apply

# Batch with different feature types
batch = {
    "continuous": torch.randn(256, 10),
    "embeddings": torch.randn(256, 50),
}

# Compute mean and std for normalization
means = mean_along_batch(batch)
stds = std_along_batch(batch)

# Normalize
def normalize_tensor(x, mean, std):
    return (x - mean) / (std + 1e-8)

normalized = {}
for key in batch:
    normalized[key] = normalize_tensor(batch[key], means[key], stds[key])

print("Normalized batch mean (should be ~0):")
print(mean_along_batch(normalized))
```

## Example 7: Debugging with Utils

Inspecting complex nested structures:

```python
import torch
from batchtensor.utils import bfs_tensor, dfs_tensor

# Complex nested batch
batch = {
    "vision": {
        "rgb": torch.randn(8, 3, 64, 64),
        "depth": torch.randn(8, 1, 64, 64),
    },
    "text": {
        "tokens": torch.randint(0, 1000, (8, 50)),
        "attention_mask": torch.ones(8, 50),
    },
    "metadata": {
        "ids": torch.arange(8),
        "timestamps": torch.randn(8),
    },
}

print("=" * 60)
print("Batch Structure (BFS):")
print("=" * 60)
for path, tensor in bfs_tensor(batch):
    path_str = ".".join(path)
    memory_mb = tensor.element_size() * tensor.numel() / (1024 ** 2)
    print(f"{path_str:30s} {str(tensor.shape):20s} {memory_mb:8.2f} MB")

print("\n" + "=" * 60)
print("Total memory usage:")
print("=" * 60)
total_memory = sum(
    tensor.element_size() * tensor.numel()
    for _, tensor in bfs_tensor(batch)
)
print(f"{total_memory / (1024 ** 2):.2f} MB")
```

## Example 8: Custom Recursive Operations

Creating custom operations with the recursive module:

```python
import torch
from batchtensor.recursive import recursive_apply

def clip_gradients(data, max_norm=1.0):
    """Clip gradients in nested structures."""
    def clip_fn(x):
        if isinstance(x, torch.Tensor) and x.requires_grad and x.grad is not None:
            torch.nn.utils.clip_grad_norm_([x], max_norm)
        return x
    
    return recursive_apply(data, clip_fn)

# Model parameters as nested dict
params = {
    "encoder": {
        "weights": torch.randn(100, 50, requires_grad=True),
        "bias": torch.randn(100, requires_grad=True),
    },
    "decoder": {
        "weights": torch.randn(50, 10, requires_grad=True),
        "bias": torch.randn(10, requires_grad=True),
    },
}

# Simulate some gradients
for _, tensor in bfs_tensor(params):
    if tensor.requires_grad:
        tensor.grad = torch.randn_like(tensor) * 10  # Large gradients

# Clip all gradients
params = clip_gradients(params, max_norm=1.0)
```

## Example 9: Data Loading Pipeline

Integrating with PyTorch DataLoader:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from batchtensor.nested import cat_along_batch

class NestedDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input": torch.randn(10),
            "target": torch.randint(0, 2, (1,)).item(),
            "metadata": {
                "id": idx,
                "weight": torch.rand(1).item(),
            },
        }

# Custom collate function
def nested_collate(batch):
    """Collate nested dictionaries."""
    # Convert list of dicts to dict of lists
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], dict):
            # Recursively collate nested dicts
            collated[key] = nested_collate(values)
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = torch.tensor(values)
    
    return collated

# Create dataloader
dataset = NestedDataset(size=100)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=nested_collate,
    shuffle=True,
)

# Use in training loop
for batch in dataloader:
    print(f"Batch input shape: {batch['input'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
    print(f"Batch metadata IDs: {batch['metadata']['id'].shape}")
    break  # Just show first batch
```

## Example 10: Validation and Testing

Batch processing for model evaluation:

```python
import torch
from batchtensor.nested import (
    chunk_along_batch,
    cat_along_batch,
    argmax_along_batch,
)

def evaluate_model(model, data, batch_size=32):
    """Evaluate model on nested data in chunks."""
    # Split large dataset into batches
    batches = chunk_along_batch(data, chunks=len(data["features"]) // batch_size)
    
    predictions = []
    
    for batch in batches:
        # Forward pass (simplified)
        with torch.no_grad():
            logits = {
                "class_logits": torch.randn(batch["features"].shape[0], 10),
            }
        predictions.append(logits)
    
    # Combine all predictions
    all_predictions = cat_along_batch(predictions)
    
    # Get predicted classes
    predicted_classes = argmax_along_batch(all_predictions)
    
    return predicted_classes

# Large test dataset
test_data = {
    "features": torch.randn(1000, 128),
    "labels": torch.randint(0, 10, (1000,)),
}

# Evaluate in batches
predictions = evaluate_model(None, test_data, batch_size=32)
print(f"Predictions shape: {predictions['class_logits'].shape}")
```

## Best Practices Summary

1. **Maintain Structure Consistency**: Keep the same nested structure throughout your pipeline
2. **Use Batch Operations**: Leverage batch operations instead of loops when possible
3. **Memory Management**: Move data to GPU only when needed, use appropriate dtypes
4. **Validation**: Use utils functions to validate batch dimensions and structure
5. **Error Handling**: Wrap operations in try-except blocks for production code
6. **Documentation**: Document the expected structure of your nested batches
7. **Testing**: Test with small batches first before scaling up

## See Also

- [Getting Started](get_started.md) - Installation and setup
- [Nested Operations Guide](uguide/nested.md) - Detailed nested operations
- [Recursive Module Guide](uguide/recursive.md) - Custom recursive operations
- [Utils Guide](uguide/utils.md) - Debugging and inspection utilities
