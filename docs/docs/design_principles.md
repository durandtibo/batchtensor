# Design principles

`batchtensor` is a set of thin wrappers around `torch` for batches whose first dimension is the
batch and whose second dimension is the sequence. The package is pre-1.0 and the API is not stable.
Each principle below is enforced by a test.

## Layout

- **Fixed layout (P1).** Batch is dim 0 and sequence is dim 1. The dims come only from
  `BATCH_DIM` and `SEQ_DIM`. No function takes a `dim` argument. Data laid out as
  `(seq, batch, ...)` must be transposed first.

## Two symmetric levels

- **Thin wrappers (P2).** A `tensor` function is one torch call with the dimension fixed. It has
  the same semantics and result as torch, with no hidden copy, cast or device move. The one
  closed list of exceptions is below. A function that is not in it is one torch call.

  | Function                                | Why it is not one torch call                                         |
  | --------------------------------------- | -------------------------------------------------------------------- |
  | `masked_{sum,mean,amax,amin}_along_seq` | `masked_fill` + reduction; the mask is validated (no sync)           |
  | `lengths_to_mask`                       | no torch equivalent; a sync only without `max_len`                   |
  | `pad_sequences`                         | no torch equivalent (`pad_sequence` is sequence-first)               |
  | `pad_along_*`                           | `F.pad` with the dim fixed; checks `value` against the dtype         |
  | `repeat_along_*`                        | builds the `repeat` sizes; checks the number of dims                 |
  | `index_select_along_*`                  | checks the index is 1-D (batch); a 2-D index is a per-sample gather (seq) |
  | `permute_*`, `shuffle_*`                | `index_select` + a length check (`shuffle_*` draws the permutation)  |
  | `topk_*`                                | option `sort_output` for `sorted` (it shadows a builtin)             |

  `test_thin_wrappers` fails for a new function that is neither one torch call nor in this list.
- **Two symmetric levels (P3).** Every `tensor` function that maps batch data has a `nested`
  twin with the same name, options and defaults. The one exception is `lengths_to_mask` (its input
  is not a batch of tensors). In `nested`, the `mask` of a masked reduction is one tensor shared
  by all the leaves. `nested` also has `clone`, `detach`,
  `contiguous`, `pin_memory`, `to`, `to_numpy` and `from_numpy`.
- **One traversal (P4).** All the nested functions go through `apply_to_tensors`,
  `split_tensors` and `combine_tensors` of `batchtensor.utils.structure`. A nested function has no
  logic of its own.

## Structure and leaves

- **Structure-preserving (P5).** A map function returns the input structure with every leaf
  replaced. Container types, keys and order follow `coola.recursive`. `cat_*`, `stack_*`,
  `chunk_*` and `split_*` also keep these types.
- **Supported containers (D1).** The supported structure is `dict`, `list`, `tuple` and their
  subclasses (named tuples, `OrderedDict`, `defaultdict`). This is exactly `NestedTensor`, and the
  three helpers of `batchtensor.utils.structure` share this rule. Anything else (sets,
  dataclasses) is outside the contract: it may work through `coola`, and no test, doc or type
  promises it.
- **Cross-leaf rule.** The functions that map a tensor function do not compare the sizes of the
  leaves. `chunk_*`, `split_*` and `shuffle_*` do, because the alignment of their output depends
  on it. `cat_*` and `stack_*` check that the structures have the same layout (keys and lengths).
- **Empty structures.** A function that does not need a size keeps the empty structure:
  `sum_along_batch({})` returns `{}`. A function that needs a size raises a `ValueError`:
  `chunk_*`, `split_*` and `shuffle_*` on a structure without tensor.
- **Tensor leaves only (P6).** A leaf that is not a `torch.Tensor` raises a `TypeError` that names
  the leaf type. Nothing is skipped silently. `from_numpy` is the only exception: its leaves
  are arrays, and the leaf type error is the `TypeError` of `torch.from_numpy`.
- **Named-tuple results.** `max`, `min`, `median` and `sort` keep their `torch.return_types.*`
  type as the leaf of the nested output.

## Behaviour

- **Pure (P7).** Inputs are never modified. The output never aliases the input, except where torch
  returns a view (`slice`, `select`, `chunk`, `split`, `detach`, `from_numpy`, `to_numpy` on CPU)
  or the input itself (`to`, `contiguous`, `pin_memory`). A function adds no shortcut that returns
  its input: `pad_along_*(x, 0)` is a new tensor.
- **One error contract (P8).** A bad leaf type is a `TypeError`. An argument that `batchtensor`
  validates is a `ValueError`. Anything else is the torch error, unchanged. `batchtensor` validates
  only what torch cannot see or what torch accepts but means something else.

  | Situation                                                           | Exception         |
  | ------------------------------------------------------------------- | ----------------- |
  | A leaf is not a `torch.Tensor`                                      | `TypeError`       |
  | `cat`/`stack` receives no item or structures with different layout  | `ValueError`      |
  | Leaves disagree on the split, chunk or shuffle dimension            | `ValueError`      |
  | `pad_size < 0`, a `value` that does not fit the dtype, a bad `side` | `ValueError`      |
  | `cat`/`stack` combines a mapping with a sequence                    | `ValueError`      |
  | `repeat_*`, `seq_len`, `shuffle_*` on a tensor with too few dims    | `ValueError`      |
  | `select_along_seq`, `slice_along_seq`, `sum_along_seq` on a 1-D tensor | torch's `IndexError` |
  | `pad_sequences` receives tensors of different dtype or device       | `ValueError`      |
  | A `permutation`/`index` with the wrong ndim or length               | `ValueError`      |
  | `lengths` is not a 1-D integer tensor                               | `ValueError`      |
  | `repeats < 0`, `chunks <= 0`, a bad dtype                           | torch's own error |

- **Explicit randomness (P9).** Randomness comes only from an explicit `generator` (a
  `torch.Generator` or an `int` seed). Without it, the global torch RNG is used. One permutation
  is shared by all the leaves.
- **Device and dtype agnostic (P10).** No function moves data between devices or casts dtype.
  There is no host-device sync, except `lengths_to_mask` without `max_len`. A function that
  would need a sync to validate documents its behaviour instead (P16).
- **Defined on edge cases (P16).** Every function has a documented result on empty and degenerate
  input. "Undefined" is not an answer; "torch's behaviour" is. `test_edge_case` pins the results:

  | Function                                  | Case                         | Result                                       |
  | ----------------------------------------- | ---------------------------- | -------------------------------------------- |
  | `masked_mean_along_seq`                   | a row without valid step     | `nan` (0 / 0)                                |
  | `masked_amax_along_seq`, `masked_amin_*`  | a row without valid step     | the lowest (highest) value of the dtype      |
  | `masked_sum_along_seq`                    | a row without valid step     | `0`                                          |
  | `permute_*`                               | a repeated index             | not checked: only the length is              |
  | `lengths_to_mask`                         | a negative length            | a row of `False`                             |
  | `lengths_to_mask`                         | `max_len` below a length     | the row is truncated to `max_len`            |
  | `pad_*`                                   | `pad_size == 0`              | a new tensor equal to the input              |
  | `pad_sequences`                           | tensors of different dtype   | `ValueError`                                 |
  | `pad_sequences`                           | a zero-length tensor         | a row of `value`                             |

  A guard against a fully masked row is `mask.any(dim=1)`, tested by the caller.
- **Containers of `cat_*`, `stack_*` and `pad_sequences`.** The container types of the output are
  the ones of the first item. `[list, tuple]` gives a list and `[dict, OrderedDict]` gives a
  dict. A mapping combined with a sequence raises a `ValueError`.
- **Differentiable (P11).** Autograd flows through every function whose torch op is
  differentiable. Only `to_numpy` detaches.

## API

- **Keyword-only options (P12).** The positional arguments are the data and the arguments that
  define the operation. Options (`keepdim`, `descending`, `stable`, `step`, `value`, `side`,
  `generator`, `max_len`, `memory_format`) are keyword-only.
- **Typed (P13).** A map function has the type `T -> T` with `T` bound to `NestedTensor`: dict in,
  dict out, tensor in, tensor out. The functions that return named tuples return `NestedTensor`.
- **Layered (P14).** `constants` <- `utils` <- `tensor` <- `nested`, enforced by `import-linter`.
  `torch` and `coola` are the only required dependencies. `numpy` is only needed for `from_numpy`
  and `to_numpy` and is imported lazily.
- **Documented by example (P15).** Every public function has a docstring with a runnable example.

## Not guaranteed

- Shared references: two keys that point to the same tensor are processed twice.
- Memory sharing between input and output, except for the views listed above.
- The order of the leaves inside a set.
- Dim-last or sequence-first layouts.
- Cross-leaf validation, except where the output alignment depends on it (`shuffle_*`, `chunk_*`,
  `split_*`).
