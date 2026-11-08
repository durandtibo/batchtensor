# Migration guide

This page lists the breaking changes of the target-state refactoring and their replacement.

## Removed functions

| Removed                                                                                                                                                               | Replacement                                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `nested.abs`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh` | `apply_to_tensors(data, torch.xxx)` from `batchtensor.utils.structure` |
| `nested.clamp(data, min, max)`                                                                                                                                        | `apply_to_tensors(data, partial(torch.clamp, min=..., max=...))`       |
| `nested.as_tensor(data)`                                                                                                                                              | `coola.recursive.recursive_apply(data, torch.as_tensor)`               |
| `nested.combine(items, fn)`                                                                                                                                           | `batchtensor.utils.structure.combine_tensors(items, fn)`               |
| `nested.split(data, fn)`                                                                                                                                              | `batchtensor.utils.structure.split_tensors(data, fn)`                  |
| `batchtensor.nested.types.NestedTensor`                                                                                                                               | `batchtensor.nested.NestedTensor` (public, excludes `str` and `bytes`). The module is now `batchtensor.utils.types`; `batchtensor.nested.types` is removed |

```pycon
>>> import torch
>>> from functools import partial
>>> from batchtensor.utils.structure import apply_to_tensors
>>> batch = {"x": torch.tensor([[-1.0, 2.0]])}
>>> apply_to_tensors(batch, torch.exp)
{'x': tensor([[0.3679, 7.3891]])}
>>> apply_to_tensors(batch, partial(torch.clamp, min=0.0))
{'x': tensor([[0., 2.]])}

```

## Options are keyword-only

`keepdim`, `descending`, `stable`, `step`, `value`, `side`, `generator`, `max_len` and
`memory_format` can no longer be passed positionally.

| Before                              | After                                         |
| ----------------------------------- | --------------------------------------------- |
| `sum_along_batch(x, True)`          | `sum_along_batch(x, keepdim=True)`            |
| `sort_along_seq(x, True)`           | `sort_along_seq(x, descending=True)`          |
| `slice_along_batch(x, 0, 4, 2)`     | `slice_along_batch(x, 0, 4, step=2)`          |
| `pad_along_seq(x, 2, 1.0)`          | `pad_along_seq(x, 2, value=1.0)`              |
| `shuffle_along_batch(x, generator)` | `shuffle_along_batch(x, generator=generator)` |
| `lengths_to_mask(lengths, 5)`       | `lengths_to_mask(lengths, max_len=5)`         |

`sort_along_*` and `argsort_along_*` no longer forward an arbitrary `**kwargs` to torch: they take
`descending` and `stable`. `nested.contiguous` takes `memory_format` instead of `*args, **kwargs`.

## Errors

| Case                                                                         | Before                                        | After                            |
| ---------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------- |
| Non-tensor leaf in a nested function                                         | `TypeError`, `AttributeError`... per function | `TypeError` naming the leaf type |
| `to_numpy` with a non-tensor leaf                                            | `AttributeError`                              | `TypeError` naming the leaf type |
| `cat_*`/`stack_*` on dataclasses                                             | `TypeError` (leaf)                            | `TypeError` that names dataclasses|
| `cat_*`/`stack_*` on an empty sequence                                       | `{}`                                          | `ValueError`                     |
| `cat_*`/`stack_*` on dicts with different keys                               | `KeyError`                                    | `ValueError`                     |
| `chunk_*`/`split_*`/`shuffle_*` on data without tensor                       | `()` / unchanged data                         | `ValueError`                     |
| `chunk_*`/`split_*` on leaves with different sizes                           | misaligned chunks, no error                   | `ValueError`                     |
| `index_select_along_seq` with a batch mismatch                               | `RuntimeError`                                | `ValueError`                     |
| `index_select_*` with an index of the wrong ndim, `permute_*` with a 2-D one | `IndexError`                                  | `ValueError`                     |
| `lengths_to_mask` with a float tensor                                        | accepted                                      | `ValueError`                     |
| `pad_along_*` with a `value` that does not fit the dtype                     | silent truncation                             | `ValueError`                     |

## Containers

`cat_*`, `stack_*`, `chunk_*` and `split_*` keep `OrderedDict`, `defaultdict` and named-tuple types
(they returned plain `dict` and `tuple`).

## New

`pad_along_*(..., side="left")`, `repeat_along_batch`, `pad_sequences`, an `int` seed for
`shuffle_*(generator=...)`, and `batchtensor.utils.structure` (`apply_to_tensors`,
`split_tensors`, `combine_tensors`, `get_size`), `nested.batch_size`, `nested.seq_len`, `nested.pad_sequences` and
`masked_{sum,mean,amax,amin}_along_seq` (`tensor` and `nested`).
