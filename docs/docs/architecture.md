# Architecture

This page describes how `batchtensor` is built. The properties it guarantees (P1 to P15) are in
[Design principles](design_principles.md). They are not repeated here.

## Layers

```text
constants          BATCH_DIM, SEQ_DIM
   ^
utils              types      NestedTensor, T
                   structure  apply_to_tensors, split_tensors, combine_tensors, get_size
                   seed       setup_torch_generator, random_permutation
   ^
tensor             one torch call with the dim fixed (+ masked_*, lengths_to_mask, pad_sequences)
   ^
nested             one helper call per function
```

A module imports only from the layers on its left in this list: `constants` <- `utils` <-
`tensor` <- `nested`. `import-linter` enforces it in CI.

## The `tensor` level

A function of `tensor` is one torch call with the dimension taken from `BATCH_DIM` or `SEQ_DIM`:

```python
def sum_along_batch(tensor: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
    return tensor.sum(dim=BATCH_DIM, keepdim=keepdim)
```

The sub-modules group the functions by operation: `slicing`, `indexing`, `joining`, `reduction`,
`comparison`, `math`, `permutation`, `masking` and `padding`.

## The `nested` level

The sub-modules of `nested` mirror the ones of `tensor`, plus `conversion` (NumPy), `misc`
(`clone`, `detach`, `to`, ...) and `size` (`batch_size`, `seq_len`). A nested function has no
logic. It is one call to one of three helpers of `batchtensor.utils.structure`, which are the only
code that traverses a nested structure:

| Shape  | Helper                       | Used by                                               |
| ------ | ---------------------------- | ----------------------------------------------------- |
| 1 -> 1 | `apply_to_tensors(data, fn)` | reductions, math, indexing, slicing, padding, masking |
| 1 -> N | `split_tensors(data, fn)`    | `chunk_*`, `split_*`                                  |
| N -> 1 | `combine_tensors(items, fn)` | `cat_*`, `stack_*`, `pad_sequences`                   |

```python
def sum_along_batch(data: T, *, keepdim: bool = False) -> T:
    return apply_to_tensors(data, partial(bt.sum_along_batch, keepdim=keepdim))
```

`apply_to_tensors` and `split_tensors` use `coola.recursive.recursive_apply`, so the container
types, keys and order are the ones of `coola`. `combine_tensors` walks several structures in
parallel because `coola` has no multi-input traversal. A function that does not fit one of the
three shapes is not a nested wrapper: it is a helper of `utils.structure`, like `get_size`.

## Adding an operation

1. Add the function to the sub-module of `tensor` and export it from `tensor/__init__.py`.
2. Add its twin to the sub-module of `nested` with the same name, options and defaults, using one
   of the three helpers, and export it from `nested/__init__.py`.
3. Add the `_along_batch` and `_along_seq` variants, with a docstring and a runnable example.
4. Add the unit tests, and update the parity and property tests if the function is an exception.

A pointwise function (`exp`, `clamp`) is not added: use `apply_to_tensors(data, torch.exp)`.

## See also

- [Design principles](design_principles.md)
- [Migration guide](migration.md)
- [Tensor](uguide/tensor.md), [Nested](uguide/nested.md), [Utils](uguide/utils.md) and
  [Constants](uguide/constants.md) guides
