# Design principles

This page describes the choices that shape `batchtensor`, what each choice gives you, and what it
costs. Knowing them helps you predict how a function behaves, and decide whether the library fits
your code or whether a plain `torch` call is a better tool for a given case.

The labels in parentheses (P1, D1, ...) are stable names for the choices. The integration tests in
`tests/integration` refer to them.

> `batchtensor` is pre-1.0. The behaviour described here is the current one, but the API is not
> guaranteed to be stable between releases.

## Layout

### The layout is fixed (P1)

The batch is dimension 0 and the sequence is dimension 1. Both come from
[`BATCH_DIM` and `SEQ_DIM`](uguide/constants.md), and no function takes a `dim` argument. A
function name says what it does and where: `sum_along_batch`, `slice_along_seq`.

- **Gain:** the call site is short and cannot pass the wrong dimension. All the functions agree on
  what "batch" and "sequence" mean, so the `tensor` and `nested` APIs stay small and symmetric.
- **Cost:** the library cannot operate on another dimension. Data laid out as `(seq, batch, ...)`
  (the default of `torch.nn.LSTM`) must be transposed first, and an operation along a feature
  dimension is a plain `torch` call.

## Two levels

### `tensor` functions are thin wrappers (P2)

A function of `batchtensor.tensor` is one torch call with the dimension fixed. It has the semantics
of that call and its result. It does not copy, cast or move data on its own. Its documentation
therefore doubles as a pointer to the matching torch function.

A small closed list of functions is more than one torch call. In each case the extra work is either
validation that prevents a silent mistake, or a combination of calls that is easy to get wrong:

| Function                                | Why it is not one torch call                                              |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `masked_{sum,mean,amax,amin}_along_seq` | `masked_fill` + reduction; the mask is validated (no sync)                |
| `lengths_to_mask`                       | no torch equivalent; a sync only without `max_len`                        |
| `pad_sequences`                         | no torch equivalent (`pad_sequence` is sequence-first)                    |
| `pad_along_*`                           | `F.pad` with the dim fixed; checks `value` against the dtype              |
| `repeat_along_*`                        | builds the `repeat` sizes; checks the number of dims                      |
| `index_select_along_*`                  | checks the index is 1-D (batch); a 2-D index is a per-sample gather (seq) |
| `permute_*`, `shuffle_*`                | `index_select` + a length check (`shuffle_*` draws the permutation)       |
| `topk_*`                                | option `sort_output` for `sorted` (it shadows a builtin)                  |

- **Gain:** no surprise. If you know the torch function, you know the result, including the dtype,
  the device and whether the output is a view.
- **Cost:** the library adds no convenience that torch does not have. For example, a reduction over
  a ragged batch needs an explicit mask (see `masked_*`); it is not inferred.

### `nested` mirrors `tensor` (P3)

Every `tensor` function that maps batch data has a `nested` function with the same name, options and
defaults. `nested` adds `clone`, `detach`, `contiguous`, `pin_memory`, `to`, `to_numpy`,
`from_numpy`, `batch_size` and `seq_len`. `lengths_to_mask` has no `nested` twin because its input
is not a batch of tensors. In the nested masked reductions, the `mask` is one tensor shared by all
the leaves.

- **Gain:** code written for a single tensor moves to a dict of tensors by changing an import.
- **Cost:** the nested level cannot express an operation that needs different arguments per leaf.
  Apply the `tensor` function to each leaf yourself for that.

### One traversal for the nested level (P4)

A nested function contains no recursion of its own. It calls one of three helpers of
[`batchtensor.utils.structure`](uguide/utils.md): `apply_to_tensors` (one structure in, one out),
`split_tensors` (one in, several out) and `combine_tensors` (several in, one out). The traversal,
the type check on the leaves and the error messages are therefore identical for all the functions.

- **Gain:** one place defines how structures are walked. You can use the same helpers for an
  operation `batchtensor` does not provide.
- **Cost:** an operation that does not fit one of the three shapes is not a nested function.
- **Consequence:** element-wise functions (`exp`, `log`, `clamp`, ...) have no nested version, since
  they do not depend on the layout. Use `apply_to_tensors(data, torch.exp)`.

## Structures and leaves

### The structure is preserved (P5)

A function that maps every leaf returns the input structure with each leaf replaced. Container types,
keys and the order of the keys are those of `coola.recursive`. `cat_*`, `stack_*`, `chunk_*` and
`split_*` also keep the container types (see below for `cat_*` and `stack_*`).

### Supported containers (D1)

The supported structure is a tensor, or a `dict`, `list` or `tuple` of supported structures, and
their subclasses (`OrderedDict`, `defaultdict`, named tuples). This is the `NestedTensor` type, and
the three structure helpers apply the same rule. Sets and dataclasses are outside the contract: they
may work through `coola`, but nothing in the tests or the documentation promises it.

- **Gain:** a small contract that is fully tested and typed.
- **Cost:** a custom container (a dataclass batch, for example) must be converted to a dict first.

### Leaves must be tensors (P6)

A leaf that is not a `torch.Tensor` raises a `TypeError` that names the type of the leaf. Nothing is
skipped silently. `from_numpy` is the one exception: its leaves are arrays.

- **Gain:** a typo or an unexpected `None` or `str` in a batch is reported where it happens, not
  later as a shape mismatch.
- **Cost:** a batch with metadata (names, ids) cannot be passed as is. Split the tensors from the
  metadata first.

### Leaves are independent, except where alignment matters

Functions that map a tensor function do not compare the sizes of the leaves. Three families check
something, because the output of different leaves must line up:

- `shuffle_*` requires all the leaves to have the same size along the dimension, since one
  permutation is shared.
- `chunk_*` and `split_*` require every leaf to be split into the same number of pieces. They do not
  require the same sizes: leaves of size 4 and 3 split by `chunks=2` give pieces of different
  lengths, and no error.
- `cat_*` and `stack_*` require all the structures to have the same layout (the same keys and
  lengths).

All of these raise a `ValueError`.

- **Gain:** the cheap functions do no extra work; the functions that would silently misalign the
  number of pieces or the permutation check.
- **Cost:** a mapped function does not tell you that two leaves have different batch sizes, and
  `chunk_*` does not detect leaves of different sizes. Check with `batch_size(data)`, which raises
  when the sizes differ.

### Empty structures

A function that does not need a size keeps the empty structure: `sum_along_batch({})` returns `{}`.
A function that needs a size raises a `ValueError`: `chunk_*`, `split_*` and `shuffle_*` on a
structure without tensor, and `cat_*` and `stack_*` on an empty sequence of structures.

### Containers of `cat_*`, `stack_*` and `pad_sequences`

The container types of the output are the ones of the first item. `[list, tuple]` gives a list and
`[dict, OrderedDict]` gives a dict. A mapping combined with a sequence raises a `ValueError`.

### Named-tuple results

`max`, `min`, `median`, `sort` and `topk` return a `torch.return_types.*` object. In the nested
output it is the leaf, so `max_along_batch({"a": x})` is `{"a": torch.return_types.max(...)}`.

## Behaviour

### Functions are pure (P7)

Inputs are never modified. The output does not alias the input, except where torch returns a view
(`slice`, `select`, `chunk`, `split`, `detach`, `from_numpy`, and `to_numpy` on CPU) or the input
itself (`to`, `contiguous`, `pin_memory`). `batchtensor` adds no shortcut that returns its input:
`pad_along_seq(x, 0)` is a new tensor.

- **Gain:** you can write to the output of a function that is not in the list above without
  touching the input.
- **Cost:** a no-op call such as `pad_along_seq(x, 0)` still allocates. Skip the call yourself if
  you know it is a no-op.

### One error contract (P8)

A leaf that is not a tensor is a `TypeError`. An argument that `batchtensor` validates is a
`ValueError`. Anything else is the torch error, unchanged. The library validates only what torch
cannot see, or what torch accepts but with another meaning than the one you intend.

| Situation                                                                    | Exception            |
| ---------------------------------------------------------------------------- | -------------------- |
| A leaf is not a `torch.Tensor`                                               | `TypeError`          |
| `cat`/`stack` receives no item or structures with different layout           | `ValueError`         |
| Leaves of different size (`shuffle_*`) or piece count (`chunk_*`, `split_*`) | `ValueError`         |
| `pad_size < 0`, a `value` that does not fit the dtype, a bad `side`          | `ValueError`         |
| `cat`/`stack` combines a mapping with a sequence                             | `ValueError`         |
| `repeat_*`, `seq_len`, `shuffle_*` on a tensor with too few dims             | `ValueError`         |
| `select_along_seq`, `slice_along_seq`, `sum_along_seq` on a 1-D tensor       | torch's `IndexError` |
| `pad_sequences` receives tensors of different dtype or device                | `ValueError`         |
| A `permutation`/`index` with the wrong ndim or length                        | `ValueError`         |
| `lengths` is not a 1-D integer tensor                                        | `ValueError`         |
| `repeats < 0`, `chunks <= 0`, a bad dtype                                    | torch's own error    |

- **Gain:** the type of the exception tells you whether the data or the arguments are wrong.
- **Cost:** two kinds of message. Torch errors keep their torch wording, and are not translated.

### Randomness is explicit (P9)

Randomness comes only from an explicit `generator` (a `torch.Generator` or an `int` seed). Without
one, the global torch random state is used. One permutation is drawn and shared by all the leaves,
so the leaves stay aligned.

### No device or dtype changes (P10)

No function moves data between devices or casts a dtype. A function does not trigger a host-device
synchronization either, except `lengths_to_mask` without `max_len` (it must read the largest
length). A function that would need a synchronization to validate its input documents its behaviour
instead of checking.

- **Gain:** the functions are safe inside a training step on an accelerator.
- **Cost:** some mistakes are not caught. `permute_*` checks the length of the permutation but not
  that it has no repeated index, because that check would synchronize.

### Edge cases have a documented result (P16)

A function never has an "undefined" result on empty or degenerate input. The result is either
documented here or "whatever torch does":

| Function                                 | Case                       | Result                                  |
| ---------------------------------------- | -------------------------- | --------------------------------------- |
| `masked_mean_along_seq`                  | a row without valid step   | `nan` (0 / 0)                           |
| `masked_amax_along_seq`, `masked_amin_*` | a row without valid step   | the lowest (highest) value of the dtype |
| `masked_sum_along_seq`                   | a row without valid step   | `0`                                     |
| `permute_*`                              | a repeated index           | not checked: only the length is         |
| `lengths_to_mask`                        | a negative length          | a row of `False`                        |
| `lengths_to_mask`                        | `max_len` below a length   | the row is truncated to `max_len`       |
| `pad_*`                                  | `pad_size == 0`            | a new tensor equal to the input         |
| `pad_sequences`                          | tensors of different dtype | `ValueError`                            |
| `pad_sequences`                          | a zero-length tensor       | a row of `value`                        |

To guard against a fully masked row, test `mask.any(dim=1)` yourself. The library does not do it
because the test needs a synchronization.

### Autograd works (P11)

Gradients flow through every function whose torch operation is differentiable. Only `to_numpy`
detaches, since a NumPy array has no graph.

## API

### Options are keyword-only (P12)

The positional arguments are the data and the arguments that define the operation. Options
(`keepdim`, `descending`, `stable`, `step`, `value`, `side`, `generator`, `max_len`,
`memory_format`) are keyword-only, so `pad_along_seq(x, 2, value=1.0)` is readable and a new option
never shifts an existing argument.

### Functions are typed (P13)

A function that maps a structure has the type `T -> T`, with `T` bound to `NestedTensor`: a dict in
gives a dict out, a tensor in gives a tensor out. The functions that return named tuples return
`NestedTensor`, because the leaf type changes.

### Layers only import downward (P14)

`constants` <- `utils` <- `tensor` <- `nested`; a module imports only from the layers on its left,
and `import-linter` checks it in CI. `torch` and `coola` are the only required dependencies. NumPy
is needed only for `from_numpy` and `to_numpy`, and it is imported lazily.

### Every function has an example (P15)

Every public function has a docstring with a runnable example that is run as a doctest.

## Not guaranteed

These behaviours exist today, but do not rely on them:

- Two keys that point to the same tensor are processed twice (shared references are not detected).
- Memory sharing between input and output, except for the views listed above.
- The order of the leaves inside a set.
- Dim-last or sequence-first layouts: they are not supported.
- Cross-leaf validation beyond the checks listed in "Leaves are independent, except where alignment
  matters".
