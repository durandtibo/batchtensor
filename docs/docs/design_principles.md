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
  the same semantics and result as torch, with no hidden copy, cast or device move.
- **Two symmetric levels (P3).** Every `tensor` function that maps batch data has a `nested`
  twin with the same name, options and defaults. The exceptions are `lengths_to_mask` and
  `pad_sequences` (their input is not a batch of tensors). `nested` also has `clone`, `detach`,
  `contiguous`, `pin_memory`, `to`, `to_numpy` and `from_numpy`.
- **One traversal (P4).** All the nested functions go through `apply_to_tensors`,
  `split_tensors` and `combine_tensors` of `batchtensor.utils.structure`. A nested function has no
  logic of its own.

## Structure and leaves

- **Structure-preserving (P5).** A map function returns the input structure with every leaf
  replaced. Container types, keys and order follow `coola.recursive`: dicts, lists, tuples, named
  tuples, `OrderedDict`, `defaultdict` and dataclasses. `cat_*`, `stack_*`, `chunk_*` and
  `split_*` also keep these types but do not support dataclasses.
- **Tensor leaves only (P6).** A leaf that is not a `torch.Tensor` raises a `TypeError` that names
  the leaf type. Nothing is skipped silently. `from_numpy` is the only exception: its leaves
  are arrays, and the leaf type error is the `TypeError` of `torch.from_numpy`.
- **Named-tuple results.** `max`, `min`, `median` and `sort` keep their `torch.return_types.*`
  type as the leaf of the nested output.

## Behaviour

- **Pure (P7).** Inputs are never modified. A function returns a view where torch returns a view
  (`slice`, `select`, `chunk`, `split`, `from_numpy`, `to_numpy` on CPU), otherwise a new tensor.
- **One error contract (P8).** A bad leaf type is a `TypeError`. An argument that `batchtensor`
  validates is a `ValueError`. Anything else is the torch error, unchanged. `batchtensor` validates
  only what torch cannot see or what torch accepts but means something else.

  | Situation                                                           | Exception         |
  | ------------------------------------------------------------------- | ----------------- |
  | A leaf is not a `torch.Tensor`                                      | `TypeError`       |
  | `cat`/`stack` receives no item or structures with different layout  | `ValueError`      |
  | Leaves disagree on the split, chunk or shuffle dimension            | `ValueError`      |
  | `pad_size < 0`, a `value` that does not fit the dtype, a bad `side` | `ValueError`      |
  | A `permutation`/`index` with the wrong ndim or length               | `ValueError`      |
  | `lengths` is not a 1-D integer tensor                               | `ValueError`      |
  | `repeats < 0`, `chunks <= 0`, a bad dtype                           | torch's own error |

- **Explicit randomness (P9).** Randomness comes only from an explicit `generator` (a
  `torch.Generator` or an `int` seed). Without it, the global torch RNG is used. One permutation
  is shared by all the leaves.
- **Device and dtype agnostic (P10).** No function moves data between devices or casts dtype.
  There is no host-device sync, except `lengths_to_mask` without `max_len`.
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
