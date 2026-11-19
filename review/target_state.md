# `batchtensor`: target state and the changes to reach it

**Scope.** This document defines the properties of `batchtensor` (`constants`, `utils`, `tensor`,
`nested`), compares them with `main` (`b602640`), and lists the changes that remain to reach the
target state. It changes no code. It replaces the previous version, whose steps 1 to 4 are merged
(#787, #788, #789, #791; Appendix B). Each finding is tagged **[verified]** (a command or a probe
shows it) or **[reading]** (found by reading the code, not run).

**Where the package stands.** The architecture is at its target and the quality gates are green on
`main`: `ruff`, `ty` and `lint-imports` are clean, 3 008 tests pass (3 skipped: no accelerator),
and every nested function is one call to one of three helpers (shuffle is the one composition). No
code of `batchtensor` walks a container. What is left is small. The one defect that changes a
result is that `chunk_*` and `split_*` return misaligned pieces, without error, when the leaves
have different sizes but split into the same number of pieces (F7). The rest is the handling of
**indices**:

1. `index_select_along_seq` with a 2-D index is not an `index_select`. It casts a float index and
   accepts negative indices, which torch refuses (F1). It also validates what torch already
   validates (F2).
2. `permute_*` moves its index to the device of the data, `index_select_*` does not (F3).
3. The documentation drifted from the code in a few places a gate can catch (F4).
4. The release is still blocked by an alpha dependency, and the metadata contradicts the version
   (F5).

This is a pre-1.0 release (`0.1.2a1`). Breaking changes are allowed when they make the package
simpler or the behaviour more predictable.

## Constraints

1. **Simple to maintain.** A rule is implemented in one place. A function is a thin wrapper. Remove
   code instead of adding a workaround.
2. **Breaking changes are allowed.** No deprecation period, no adapter. The future state matters
   more than the past state.
3. **Consistent with `coola`.** Container handling (types, keys, order) is the one of
   `coola.recursive`. `batchtensor` does not re-implement it.
4. **Follow `CLAUDE.md`.** Batch is dim 0, sequence is dim 1, `tensor` and `nested` stay in sync,
   options are keyword-only, shared helpers are public functions of `batchtensor.utils`, optional
   libraries are imported lazily.
5. **No slower than today.** A nested call is dominated by the torch call (Section 6.1).
6. **A claim has a gate.** A property written in the docs is enforced by a test or a CI job that
   fails when it breaks. A property without a gate is given one or removed.
7. **No hidden synchronisation.** A check that needs the values of a tensor (a host-device sync) is
   not added to a function. It is documented instead (P10, P16).
8. **No hidden conversion.** A function does not cast, move or reinterpret what the caller passes,
   whether it is data or an index. It forwards it to torch and lets torch decide (P8, P10).

## 1. Properties of the package

These are the properties the package guarantees by construction. Each has an owner (the code that
enforces it) and a gate (the test or job that fails when it breaks). `P2`, `P8` and `P10` change
(the change is in the statement). `P17` is new.

| #   | Property                  | Statement                                                                                                                                                                                                                                                                                                                       | Owner                           | Gate                                                                                 |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| P1  | Fixed layout              | Batch is dim 0, sequence is dim 1. The dims come only from `BATCH_DIM` and `SEQ_DIM`. No function of `tensor` or `nested` takes a `dim`. `utils.structure` is dim-generic (the layer below).                                                                                                                                    | `constants.py`                  | `test_no_dim_parameter`                                                              |
| P2  | Thin wrappers             | A `tensor` function is one torch call with the dim fixed: same semantics, same result, no hidden copy, cast or device move. The **closed** list of exceptions is in Section 3.4 (a fixed composition, or a check torch does not make). A function not in the list is a thin wrapper.                                            | `tensor/*`                      | `test_thin_wrappers`, `test_torch_reference`                                         |
| P3  | Two symmetric levels      | Every `tensor` function that maps batch data has a `nested` twin with the same name, options and defaults. The exception is listed in the parity test.                                                                                                                                                                          | Parity test                     | `test_api_parity`                                                                    |
| P4  | One traversal             | Every nested function goes through `apply_to_tensors`, `split_tensors` or `combine_tensors`. No nested function walks a container. `from_numpy` (array leaves) calls `recursive_apply` and is the one listed exception.                                                                                                         | `utils/structure.py`            | `test_nested_module_does_not_walk_containers`                                        |
| P5  | Structure-preserving      | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple`, named tuple, `OrderedDict`, `defaultdict`. For `cat_*`, `stack_*` and `pad_sequences` the container types of the output are the ones of the first item.                                                     | `apply_to_tensors`              | `test_containers`, `test_differential`                                               |
| P6  | Tensor leaves only        | A leaf that is not a `torch.Tensor` raises a `TypeError` that names the leaf type. Nothing is skipped silently. Only `from_numpy` takes other leaves.                                                                                                                                                                           | `apply_to_tensors`              | `test_non_tensor_leaf`                                                               |
| P7  | Pure                      | Inputs are never modified. The output never aliases the input, except where torch returns a view or the input itself (`to`, `contiguous`, `pin_memory`; Section 3.4). A function adds no shortcut that returns its input.                                                                                                       | Thin wrappers                   | `test_inputs_are_not_modified`, `test_output_does_not_alias_input`                   |
| P8  | One error contract        | A bad leaf type is a `TypeError`. An argument that `batchtensor` validates is a `ValueError`. Anything else is the torch error, unchanged. **`batchtensor` validates only what torch cannot see or what torch accepts but means something else. It never repeats a check torch makes.** The table is in `design_principles.md`. | Validators in `utils`, `tensor` | `test_error_contract` (one row per line of the table)                                |
| P9  | Explicit randomness       | Randomness comes only from `generator` (a `torch.Generator` or an `int`). Without it, the global torch RNG. One permutation is shared by all leaves.                                                                                                                                                                            | `utils.seed`                    | `test_seed`, `test_device`                                                           |
| P10 | Device and dtype agnostic | **No function moves or casts anything the caller passes, data or index.** The one move is the permutation that `shuffle_*` draws itself (a generator lives on the CPU, the data may not). No host-device sync, except `lengths_to_mask` without `max_len`. A function that would need a sync documents instead (P16).           | Thin wrappers                   | `test_device` (skipped without accelerator), `test_output_dtype_is_the_one_of_torch` |
| P11 | Differentiable            | Autograd flows through every function whose torch op is differentiable. Only `to_numpy` and `detach` cut the graph.                                                                                                                                                                                                             | Thin wrappers                   | `test_autograd`                                                                      |
| P12 | Keyword-only options      | Positional arguments are the data and the arguments that define the operation. Options are keyword-only. No `**kwargs` forwarding, except `nested.to`. A torch option is renamed only when it shadows a builtin (`sorted` -> `sort_output`).                                                                                    | Signatures                      | `test_signatures`                                                                    |
| P13 | Typed                     | A map function has the type `T -> T`, `T` bound to `NestedTensor`. Checked by `ty` and `pyright`.                                                                                                                                                                                                                               | Annotations                     | `ty` and `pyright` in CI, `test_typing`                                              |
| P14 | Layered                   | `constants` <- `utils` <- `tensor` <- `nested`. No import in the other direction. `torch` and `coola` are the only required dependencies. `numpy` is lazy.                                                                                                                                                                      | `import-linter`                 | `ci-cyclic-imports`                                                                  |
| P15 | Documented once           | Every public function has a docstring with a runnable `pycon` example. Each rule is stated in one place (`design_principles.md`) and not repeated in the function docstrings.                                                                                                                                                   | Docstrings, docs                | `ci-doctest`, `test_docstring_does_not_repeat_layout`                                |
| P16 | Defined on edge cases     | Every function has a documented result on empty and degenerate input: a structure without tensor, a zero-length dim, a fully masked row, a zero pad. "Undefined" is not an answer; "torch's behaviour" is.                                                                                                                      | Docstrings (`Note`/`Returns`)   | `test_edge_case`                                                                     |
| P17 | Documented claims exist   | **Every file, workflow or module that the README and the docs point to exists, and the description of a module matches its content.**                                                                                                                                                                                           | Docs                            | **New: `test_docs_references` (T12)**                                                |

**Not guaranteed:**

- Shared references. `clone({"a": x, "b": x})` returns two tensors.
- Memory sharing between input and output, except for the views listed in Section 3.4.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`shuffle_*`, `chunk_*`,
  `split_*`).
- 0-d tensors. `sum_along_batch(torch.tensor(1.))` returns `tensor(1.)` because torch accepts
  `dim=0` on a scalar. It is torch's behaviour (P8) and has no batch meaning.
- Dataclasses and sets (D1). They work today through `coola` in all five function families. No test,
  doc or type promises it.
- That an index or a `permutation` argument is a valid permutation (D6).
- A generator or a lazy iterable as `items` of `cat_*`, `stack_*` and `pad_sequences`. They take a
  `Sequence`; the error of any other input is the one of torch or `coola`.

## 2. What differs today

Surface: 72 names in `tensor`, 81 in `nested` (80 functions and `NestedTensor`). The previous
findings are closed (Appendix B). The findings below are new. They are small. F1 and F7 are the two
that change what a function returns.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Tag        | Property            |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------- |
| F1  | **`index_select_along_seq` with a 2-D index casts and wraps silently.** A 2-D index runs `tensor[batch_index, index.flatten().long()]`. `index_select_along_seq(x, tensor([[0., 1.], [1., 2.]]))` returns rows (the float index is truncated by `.long()`) and `tensor([[0, -1], ...])` returns the last position. The 1-D path (`torch.index_select`) rejects both: `RuntimeError: Expected dtype int32 or int64` and `IndexError`. `torch.gather` rejects both too. The name `index_select` promises torch's rules; the 2-D branch is a different function (a per-sample gather) that applies others.                                                                                  | [verified] | P8, P10             |
| F2  | **`index_select_*` and `permute_*` repeat a check torch makes.** `index_select_along_batch(x, ones(2, 2))` raises `ValueError` from `batchtensor`, while torch raises `IndexError: Index is supposed to be a vector` on the same call. P8 says `batchtensor` validates what torch cannot see. The `ndim` check of `index_select_*` is redundant and makes the exception type differ from torch's. The length check of `permute_*` is not redundant (torch accepts a shorter index and means something else).                                                                                                                                                                             | [verified] | P2, P8              |
| F3  | **Index device handling differs between sibling functions.** `permute_*` runs `index=permutation.to(tensor.device)`; `index_select_*` and the 2-D branch of `index_select_along_seq` pass the index as is, so a CPU index with CUDA data raises. P10 allows "an index may be moved", which licenses the difference. The move exists only because `shuffle_*` draws its permutation on the generator's device (the CPU for an `int` seed) and the data may be elsewhere. It belongs to `shuffle_*`.                                                                                                                                                                                       | [reading]  | P2, P10             |
| F4  | **The docs point to things that do not exist or describe a module that changed.** The README and `docs/docs/index.md` link the badges of `nightly-tests.yaml`, `docs.yaml` and `docs-dev.yaml`; the workflows are `nightly-test-package.yaml` and `release-docs*.yaml`. A `black` badge stays although the project formats with `ruff`. `index.md` describes `batchtensor.utils` as "Utility functions for seed management" and `uguide/utils.md` says it "currently contains seed management utilities", while `utils.structure` is the module every nested function uses.                                                                                                              | [verified] | P15, P17            |
| F5  | **The release is blocked by an alpha dependency and the metadata contradicts the version.** `dependencies = ["coola>=2.0.0a4,<3.0", ...]`: `pip` skips pre-releases, so users need `--pre`. PyPI has `coola` 1.1.10 as the latest stable and `2.0.0` only as `a0` to `a4`. `batchtensor` on PyPI is `0.1.1` (classifier `3 - Alpha`); `main` is `0.1.2a1` with the classifier `4 - Beta`.                                                                                                                                                                                                                                                                                                | [verified] | P14                 |
| F6  | **Dataclasses and sets work everywhere but are out of the contract.** Nothing tests that they work or fail: a `coola` change could break users and no gate would notice. This is accepted (D1); it is listed so it is not mistaken for a promise.                                                                                                                                                                                                                                                                                                                                                                                                                                        | [verified] | P5                  |
| F7  | **`chunk_*` and `split_*` accept leaves of different sizes and return misaligned pieces.** `nested.chunk_along_batch({"a": arange(5), "b": arange(3)}, 2)` returns `({"a": [0, 1, 2], "b": [0, 1]}, {"a": [3, 4], "b": [2]})`: row 3 of `a` is paired with row 2 of `b`. `split_along_batch({"a": arange(5), "b": arange(3)}, 2)` raises only because the leaves give 3 and 2 pieces. `split_tensors` checks the **number of pieces**, not the **size**. `design_principles.md` says "`chunk_*`, `split_*` and `shuffle_*` do compare the sizes", and `test_error_contract` pins sizes whose piece counts differ, so the gap is invisible. `shuffle_*` is correct (it calls `get_size`). | [verified] | P5, cross-leaf rule |

**Checked and found correct** (no action): `pad_sequences` raises on mixed dtype, zero-d, and
mismatched trailing shape, and keeps `requires_grad`; `pad_*(x, 0)` is a new tensor;
`masked_mean_along_seq` on a fully masked row is `nan` and its gradient is `0`, not `nan`;
`masked_amax_along_seq` propagates `nan` and keeps `-inf`; `nested.cat_along_batch([{"a": x}])`
returns a new tensor; `cat_*` on a tuple of dicts works; a non-tensor leaf and a `str` raise a
`TypeError` that names the type; `max`, `min`, `median`, `sort` and `topk` keep their
`torch.return_types.*` leaves; empty structures follow the rule of `design_principles.md`.

**Behaviours that must not regress:** the container types of the input are kept (named tuple,
`OrderedDict`, `defaultdict`); one permutation is shared by all leaves of a `shuffle_*`; `chunk_*`
and `split_*` raise on leaves of different sizes (F7 makes this true); a non-tensor leaf raises a
`TypeError` that names the type; a cycle raises `ValueError`; the doctests run.

## 3. Architecture

The architecture stays. It has three layers and one traversal module.

```text
constants          BATCH_DIM, SEQ_DIM
   ^
utils              types      NestedTensor, T
                   structure  apply_to_tensors, split_tensors, combine_tensors, get_size
                   seed       setup_torch_generator, random_permutation
   ^
tensor             one torch call with the dim fixed (+ the closed list of Section 3.4)
   ^
nested             one helper call per function: partial(bt.f, **options)
```

### 3.1 A nested function is logic-free

```python
def sum_along_batch(data: T, *, keepdim: bool = False) -> T:
    return apply_to_tensors(data, partial(bt.sum_along_batch, keepdim=keepdim))
```

Three shapes exist, and only three. A new operation uses one of them:

| Shape  | Helper                       | Used by                                               |
| ------ | ---------------------------- | ----------------------------------------------------- |
| 1 -> 1 | `apply_to_tensors(data, fn)` | reductions, math, indexing, slicing, padding, masking |
| 1 -> N | `split_tensors(data, fn)`    | `chunk_*`, `split_*`                                  |
| N -> 1 | `combine_tensors(items, fn)` | `cat_*`, `stack_*`, `pad_sequences`                   |

A function that does not fit one of these shapes is not a nested wrapper and does not belong in
`nested` (it belongs in `utils.structure` as a helper, like `get_size`). `shuffle_*` is the one
composition: `get_size` + `random_permutation` + `permute_*`, so the permutation is shared.

An argument that is data of the operation but not a batch (a `mask`, an `index`, a `permutation`)
is one tensor shared by all the leaves. It is a keyword of the `partial`, not a leaf.

### 3.2 The container rule (D1)

The supported structure is `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). This is exactly `NestedTensor`. For the N -> 1 helpers the output
containers are the ones of the first item; the other items must have the same keys and lengths.
Sets and dataclasses are outside the contract (F6).

### 3.3 One statement of each rule

- The layout (P1), the container rule, the cross-leaf rule, the empty-structure rule and the error
  table are in `design_principles.md` and nowhere else. The module docstrings of `tensor` and
  `nested` link to it.
- A function docstring states only what is specific to the function: its edge cases (P16), its
  `Raises`, and its example.

### 3.4 The closed list of exceptions to P2 and P7, after the change

A function outside this list is one torch call with the dim fixed, and returns a new tensor or the
view torch returns. Compared with `main`: `index_select_along_*` leaves the list (F1, F2), `permute_*`
keeps only a length check (F3), and `gather_along_seq` enters it.

| Function                                | Why it is not one torch call                                              |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `masked_{sum,mean,amax,amin}_along_seq` | `masked_fill` + reduction; mask validated (no sync)                       |
| `lengths_to_mask`                       | no torch equivalent; a sync only without `max_len`                        |
| `pad_sequences`                         | no torch equivalent (`torch.nn.utils.rnn.pad_sequence` is seq-first)      |
| `pad_along_*`                           | `F.pad` with the dim fixed; checks `value` against the dtype              |
| `repeat_along_*`                        | builds the `repeat` sizes; checks the number of dims                      |
| `gather_along_seq` (new)                | `torch.gather` with the index reshaped and expanded over the feature dims |
| `permute_*`                             | `index_select` + a length check (torch would accept a shorter index)      |
| `shuffle_*`                             | draws the permutation, moves it to the data device, then `permute_*`      |
| `topk_*`                                | option `sort_output` for `sorted` (builtin shadowing, P12)                |

Views (aliasing allowed): `slice_*`, `select_*`, `chunk_*`, `split_*`, `detach`, `from_numpy`,
`to_numpy` on CPU. Returns the input when nothing changes (torch's rule): `to`, `contiguous`,
`pin_memory`. Everything else returns a new tensor.

### 3.5 Indices: one rule (F1, F2, F3)

An index argument is handled the same way in every function:

| Function                 | Index                         | Checked by `batchtensor`           | Moved or cast                           |
| ------------------------ | ----------------------------- | ---------------------------------- | --------------------------------------- |
| `index_select_along_*`   | 1-D `long` (or `int`) tensor  | nothing: torch's errors            | never                                   |
| `permute_*`              | 1-D tensor of length = dim    | `ndim` and length (ValueError)     | never                                   |
| `shuffle_*`              | drawn by `random_permutation` | n/a                                | moved to the data device (the one move) |
| `gather_along_seq` (new) | 2-D `long` tensor `(B, L)`    | `ndim` and batch size (ValueError) | never                                   |

`gather_along_seq(tensor, index)` returns `out[b, l] = tensor[b, index[b, l]]` with shape
`(B, L, *tensor.shape[2:])`. It is `torch.gather(tensor, 1, index_expanded)`, so the dtype and the
bounds (including negative values) are torch's rules. There is no `gather_along_batch`: a
per-sample gather along the batch dimension has no meaning, and `index_select_along_batch` is the
batch-level operation (admission rule 3, Section 3.7). A typical use is the last valid step of a
padded batch: `gather_along_seq(x, (lengths - 1).unsqueeze(1))`.

### 3.6 Edge cases: document, do not guess (P16, constraint 7)

When a check needs the values of a tensor, the function does not make it. The docstring states the
result instead, and `test_edge_case` pins it. The table is unchanged and is in `design_principles.md`:
a fully masked row (`masked_mean` -> `nan`, `masked_amax`/`amin` -> the lowest/highest value of the
dtype, `masked_sum` -> `0`), a repeated `permute_*` index (not checked), a negative length in
`lengths_to_mask` (a row of `False`), `max_len` below a length (truncated), `pad_size == 0` (a new
tensor), `pad_sequences` on a zero-length tensor (a row of `value`).

### 3.7 Admission rule for a new operation

An operation enters `tensor` and `nested` only if all of these hold. It is in `CONTRIBUTING.md` and
`architecture.md`.

1. It needs the fixed dim (`torch.f(x, dim=BATCH_DIM)`), or it composes two torch calls in a way
   that is easy to get wrong (`masked_*`, `pad_sequences`). A pointwise function
   (`exp`, `clamp`) does not: use `apply_to_tensors(data, torch.exp)`.
2. It has a `nested` twin of shape 1 -> 1, 1 -> N or N -> 1 (Section 3.1).
3. It comes with the four functions (batch and seq, tensor and nested) or a stated reason why only
   one dim makes sense (as `masked_*_along_seq`, `gather_along_seq`).
4. If it is not one torch call, it is added to the list of Section 3.4.
5. It has a runnable example, a documented edge-case result (P16), and a gate for each property it
   touches.

## 4. Public API after the change

One function is added (`gather_along_seq`, both levels). Nothing is removed. Changes:

| Name                                                 | Change                                                                                                                                                                                                                                                                       | Breaking                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `index_select_along_seq`                             | 1-D index only. A 2-D index raises torch's `IndexError`. It is one `tensor.index_select(dim=SEQ_DIM, index=index)` (F1).                                                                                                                                                     | Yes                       |
| `index_select_along_batch`                           | The `ndim` check is removed: a non-1-D index raises torch's `IndexError`, not a `ValueError` (F2). A 0-d index is accepted as in torch.                                                                                                                                      | Yes (rare)                |
| `chunk_*`, `split_*` (`nested`)                      | Raise a `ValueError` when the leaves have different sizes along the dimension, whatever the number of pieces (F7). `split_tensors(data, fn, *, dim)` calls `get_size(data, dim)` first, as `shuffle_*` does; the check of the piece counts becomes redundant and is removed. | Yes (it was a silent bug) |
| `gather_along_seq` (new, both levels)                | Per-sample gather on `torch.gather`. In `nested` the `index` is one tensor shared by all the leaves, like a `mask`.                                                                                                                                                          | No                        |
| `permute_*`                                          | The permutation is not moved to the device of the data (F3). A CPU permutation with CUDA data raises torch's error.                                                                                                                                                          | Yes (rare)                |
| `shuffle_*`                                          | Moves the permutation it draws to the device of the data. Same behaviour as today.                                                                                                                                                                                           | No                        |
| `design_principles.md`                               | P2 list (Section 3.4), P8 sentence "never repeats a check torch makes", P10 wording, P17, the error table rows for `index_select`.                                                                                                                                           | No                        |
| `README.md`, `docs/docs/index.md`, `uguide/utils.md` | Badges, module description and overview fixed (F4).                                                                                                                                                                                                                          | No                        |
| `pyproject.toml`                                     | `coola>=2.0,<3.0` when stable (F5); classifier and version set together.                                                                                                                                                                                                     | No                        |

## 5. Dependents

| Package             | Change                                                                                                                                                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola`             | Nothing to request. Left: a stable `2.0` release (F5).                                                                                                                                                                              |
| `batchtensor` users | Code that calls `index_select_along_seq` with a 2-D index uses `gather_along_seq` (and passes an integer index). Code that passes a CPU permutation to `permute_*` with accelerator data moves it first (`shuffle_*` is unchanged). |

## 6. Evidence

### 6.1 Cost

Probes of the previous review (Python 3.14, CPU, 200 keys x 5 tensors of shape `(4, 3)` = 1 000
leaves). **Not re-measured**: no change since then touches the traversal (`#785` moved
`combine_tensors` to `coola`).

| Call                                              | Time per call |
| ------------------------------------------------- | ------------- |
| `nested.sum_along_batch(data)`                    | 2.3 ms        |
| hand-written dict/list comprehension, same result | 1.1 ms        |
| `nested.chunk_along_batch(data, 2)`               | 2.6 ms        |
| `nested.cat_along_batch([data, data])`            | 2.1 ms        |

The overhead is about 1.2 microseconds per leaf (about 2x the comprehension on tiny tensors) and
negligible on real batches. Constraint 5 holds.

### 6.2 Probes of behaviour (this review, `main` at `b602640`, torch 2.15, CPU)

| Probe                                                                                    | Result                                                                 |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `index_select_along_seq(x, tensor([[0., 1.], [1., 2.], [0., 0.]]))`                      | works, float index truncated (F1)                                      |
| `index_select_along_seq(x, tensor([[0, -1], [1, 2], [0, 0]]))`                           | last position of row 0, no error (F1)                                  |
| `index_select_along_batch(x, tensor([-1]))`                                              | torch's `IndexError`                                                   |
| `x.gather(1, float index)`, `x.gather(1, index with -1)`                                 | `RuntimeError` dtype, `RuntimeError` out of bounds (F1)                |
| `x.index_select(0, tensor([[0, 1]]))`, `x.index_select(0, tensor(1))`                    | `IndexError: Index is supposed to be a vector`; shape `(1, 3)` (F2)    |
| `permute_along_batch(x, int32 index)`, `index_select_along_batch(x, int32)`              | both work                                                              |
| `masked_mean_along_seq`, fully masked row, `float16`                                     | `nan`; gradient `0`                                                    |
| `masked_amax_along_seq`, row of `nan` / row of `-inf`                                    | `nan` / `-inf`                                                         |
| `masked_amax_along_seq` / `masked_sum_along_seq` on `bool`                               | works (`bool` / `int64` output, torch's rule)                          |
| `pad_sequences` with a 0-d tensor, mismatched trailing shape or `ndim`                   | `ValueError`                                                           |
| `pad_sequences([ones(2, requires_grad=True), ones(3)]).requires_grad`                    | `True`                                                                 |
| `lengths_to_mask(tensor([]))`, `lengths_to_mask(tensor([1]), max_len=-1)`                | shape `(0, 0)`; torch's `RuntimeError`                                 |
| `repeat_along_batch(x, 0)`, `repeat_along_batch(x, -1)`                                  | shape `(0, 4)`; torch's `RuntimeError`                                 |
| `nested.cat_along_batch([{"a": x}])["a"] is x`                                           | `False`                                                                |
| `nested.cat_along_batch` with a generator                                                | `TypeError: object of type 'generator' has no len()` (out of contract) |
| `nested.sum_along_batch({})`, `nested.batch_size({})`, `nested.chunk_along_batch({}, 2)` | `{}`; `ValueError`; `ValueError`                                       |
| `nested.sum_along_batch({x})` (set), `({"a": (x, x)})`                                   | work (D1)                                                              |
| `nested.sum_along_batch(["a"])`                                                          | `TypeError` naming `str`                                               |
| `nested.chunk_along_batch` on leaves of size 5 and 3, and of size 4 and 5                | works, misaligned pieces (F7)                                          |
| `nested.split_along_batch` on leaves of size 5 and 3, `split_size=2`                     | `ValueError` on the piece counts `[2, 3]` (F7)                         |

Commands: `ruff check src tests` clean, `ty check` clean, `lint-imports` 1 contract kept,
`pytest tests` 3 008 passed, 3 skipped. No accelerator was available, so the device gates did not
run and F3 is **[reading]**.

## 7. Changes to reach the target state

One pull request per step, each releasable. Steps 1 to 4 are independent.

| Step | Change                                                                                                                                                                                                                                                                                                    | Breaking   | Status                          |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------- |
| 1    | Add `gather_along_seq` (`tensor` and `nested`) on `torch.gather`. Restrict `index_select_along_seq` to a 1-D index, remove the `ndim` check of `index_select_*` (F1, F2). Regression tests for the float and the negative index fail without the fix. Update P2 list, P8 rows, `uguide`, migration guide. | Yes        | Ready                           |
| 2    | `permute_*`: remove `.to(tensor.device)`; `shuffle_*`: move the permutation it draws (F3). Add `test_permute_does_not_move_the_index` (meta or mock device) and keep `test_shuffle_with_a_cpu_generator`. Run `test_device` on an accelerator before merging.                                             | Yes (rare) | Ready, needs an accelerator run |
| 3    | Fix the badges, the module descriptions and the `uguide/utils.md` overview (F4). Add `test_docs_references` (T12): every `actions/workflows/<name>.yaml` linked in the README and `docs/docs/*.md` exists in `.github/workflows`, every module in `refs/*.md` exists.                                     | No         | Ready                           |
| 4    | `split_tensors(data, fn, *, dim)`: validate the size with `get_size`, remove the check of the piece counts; `nested.chunk_*` and `nested.split_*` pass `dim` (F7). Regression tests with sizes `(5, 3)` and `(4, 5)` fail without the fix; `test_error_contract` rows use sizes with equal piece counts.  | Yes        | Ready                           |
| 5    | When `coola` 2.0 is stable: depend on `coola>=2.0,<3.0` (F5).                                                                                                                                                                                                                                             | No         | Blocked: no stable `coola` 2.0  |
| 6    | Set the classifier `Development Status` and the version to the first non-alpha release. Remove "the API is not stable" from `design_principles.md` and `index.md` only when D3 is answered.                                                                                                               | No         | Blocked: step 5 and D3          |

Steps 1 to 4 can ship now. Step 5 blocks the release. Step 4 is the most urgent: it is the only
one that fixes a wrong result.

## 8. Tests and gates

- **Kept:** `test_no_dim_parameter`, `test_nested_module_does_not_walk_containers`,
  `test_thin_wrappers`, `test_output_does_not_alias_input`,
  `test_output_dtype_is_the_one_of_torch`, `test_inputs_are_not_modified`, `test_edge_case`,
  `test_api_parity`, `test_signatures`, `test_differential`, `test_torch_reference`,
  `test_autograd`, `test_device`, `test_non_tensor_leaf`, `test_error_contract`, `test_containers`,
  `test_typing` (`ty` and `pyright`), doctests. 100% coverage of the code added or changed. A bug
  fix adds a regression test that fails without the fix (`git stash push src`).
- **T13. Index rule (new).** `index_select_along_*` and `gather_along_seq` with a float index and
  with a negative index raise (torch's error). One parametrized test over the table of Section
  3.5. Gate for F1.
- **T14. `gather_along_seq` reference (new).** Equal to `torch.stack([x[b, index[b]] ...])` on
  random data (seeded), for 2-D, 3-D and 4-D `x`; autograd flows; the nested twin shares one index
  between leaves of different feature shapes. Gate for P2, P3, P11.
- **T15. Permute does not move (new).** A permutation on another device than the data raises torch's
  error in `permute_*` and works in `shuffle_*` (`test_device`, accelerator; a `meta`-device
  variant runs everywhere). Gate for F3.
- **T16. Misaligned chunk (new).** `chunk_*` and `split_*` raise on leaves of size `(5, 3)` and
  `(4, 5)` with equal piece counts, at both levels. Gate for F7; replaces the weaker rows of
  `test_error_contract`.
- **T12. Docs references (new).** See step 3. Gate for F4 and P17.
- **No test** says that sets or dataclasses are supported (D1).

## 9. Decisions for the maintainer

| #   | Decision                                                                                  | Recommendation                                                                                                                                                                                                                                                                                                                             |
| --- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1  | Dataclasses and sets: supported or outside the contract?                                  | **Outside the contract.** They work through `coola` and a user can rely on it at their own risk. A promise needs a type (`NestedTensor` cannot express a dataclass) and a gate in five families, for a use case where batches are dicts.                                                                                                   |
| D2  | `index_select_along_seq` with a 2-D index: keep or split? (was open)                      | **Split, now a bug fix and not a taste.** The 2-D branch casts a float index and wraps a negative one (F1). `gather_along_seq` on `torch.gather` gets torch's rules for free and removes `index_select_along_*` from the exception list. Alternative: keep it and add the two checks; it adds validation code and keeps a misleading name. |
| D3  | When is the API stable?                                                                   | After step 5 and one release with no change. Until then keep "pre-1.0".                                                                                                                                                                                                                                                                    |
| D4  | Fully masked rows: raise, return a sentinel, or return `nan`?                             | **Document the current result (Section 3.6).** A check needs a sync (constraint 7). A user who needs a guard tests `mask.any(dim=1)`. Alternative: an option `empty_value=` on amax/amin; it adds an option to four functions for a case the caller can handle.                                                                            |
| D5  | `pad_sequences` with mixed dtype: raise or promote like `torch.cat`?                      | **Raise** (done, #787). One rule (P10: no cast) and no promotion table to maintain.                                                                                                                                                                                                                                                        |
| D6  | `permute_*`: validate that the index is a permutation?                                    | **No.** It needs a sort or a scatter and a sync. Keep the length check and document it. `shuffle_*` always creates a valid permutation.                                                                                                                                                                                                    |
| D7  | Generate the nested twins instead of writing them?                                        | **No.** Generation hides the signature from the type checkers and the docs, and a nested function is already one line. The cost is docstrings (about 30 lines each), which are the product.                                                                                                                                                |
| D8  | Add pointwise nested functions back (`exp`, `clamp`)?                                     | **No.** `apply_to_tensors(data, torch.exp)` is the API (migration guide).                                                                                                                                                                                                                                                                  |
| D9  | Drop the `tensor` level from the public docs and keep only `nested`?                      | **No.** `tensor` is what people use on a single tensor, and `nested` is defined by it (P2, P3).                                                                                                                                                                                                                                            |
| D10 | Make `cat_*` strict on the kind of container (all items the same type)?                   | **No.** It needs a check of `batchtensor` on top of `coola`'s. The first-item rule is documented and pinned.                                                                                                                                                                                                                               |
| D11 | `permute_*`: keep moving the index to the data device, or move only in `shuffle_*`? (new) | **Move only in `shuffle_*`** (F3). One rule for every index (Section 3.5) and `permute_*` becomes `index_select` + a length check. Alternative: keep the move in `permute_*` and add it to `index_select_*` and `gather_along_seq`; three functions then move data-like arguments, against constraint 8.                                   |
| D12 | Add `unbind_along_batch` or a `masked_{prod,std,var}`? (new)                              | **No.** None needs the fixed dim more than `select_along_batch` or `masked_*` do, and each costs four functions (Section 3.7). `gather_along_seq` passes because its only alternative is the buggy branch it replaces. Revisit on a request with a use case.                                                                               |

## 10. Not covered

- New operations beyond `gather_along_seq`. The admission rule (Section 3.7) is the process (D12).
- Ragged or jagged data (`torch.nested`), sparse tensors, `torch.compile` support.
- A `dim` argument or a configurable layout (P1 says no).
- Data loaders, datasets or collate functions (`pad_sequences` and `cat_*` are the building
  blocks, not a framework).
- Python and `torch` version policy beyond `torch>=2.4,<3.0` and Python 3.11 to 3.15.
- Performance work. Section 6.1 shows no need.
- Behaviour on an accelerator. The device gates were not run (no accelerator on the review
  machine); a CUDA run of `test_device` is required before step 2 and before step 6.
- Annotation of `cat_*`/`stack_*` (`list | tuple`, torch's own contract) versus `pad_sequences`
  (`Sequence`): both are accurate for what each calls.

## Appendix A. API inventory

**`tensor` (72 names, 73 with `gather_along_seq`).** Reductions (`amax`, `amin`, `argmax`,
`argmin`, `max`, `mean`, `median`, `min`, `prod`, `sum`, `std`, `var`, `any`, `all`, `logsumexp`),
cumulative (`cumsum`, `cumprod`), sorting (`sort`, `argsort`, `topk`), indexing (`select`,
`index_select`, `gather_along_seq`), slicing (`slice`, `chunk`, `split`), joining (`cat`, `stack`,
`repeat`), permutation (`permute`, `shuffle`, `flip`, `roll`), padding (`pad`), each `_along_batch`
and `_along_seq`; plus `masked_{sum,mean,amax,amin}_along_seq`, `lengths_to_mask`, `pad_sequences`.

**`nested` (80 functions, 81 with `gather_along_seq`).** The same names except `lengths_to_mask`,
plus `clone`, `detach`, `contiguous`, `pin_memory`, `to`, `to_numpy`, `from_numpy`, `batch_size`,
`seq_len`, and the type `NestedTensor`.

**`utils`.** `structure` (`apply_to_tensors`, `split_tensors`, `combine_tensors`, `get_size`),
`seed` (`setup_torch_generator`, `random_permutation`), `types` (`NestedTensor`, `T`).

**`constants`.** `BATCH_DIM`, `SEQ_DIM`.

**Size.** `src`: 7 053 lines (the nested docstrings average 30 lines per function); `tests`:
10 221 lines. Every public `nested` function body is one `return`, except `shuffle_*` (two
statements, the one composition).

## Appendix B. Record of the previous target states

The first version listed 11 steps, all merged: the three structure helpers, keyword-only options,
removal of the pointwise nested functions, the validation contract, `nested.batch_size` and
`seq_len`, `pad_sequences`, the masked reductions, `pyright` in CI, the gates for P1, P4 and P7,
and the migration guide.

The second version (steps 1 to 6 merged): `architecture.md` rewritten from the properties (#777),
the layout `Note:` blocks removed (#778), the container, cross-leaf and empty-structure rules
(#779), the error contract test (#780), the admission rule (#781), `coola>=2.0.0a3` and the
`rebuild_*` import from `coola.recursive` (#782), and `combine_tensors` on
`coola.recursive.recursive_zip_apply` (#785).

The third version (steps 1 to 4 merged): `pad_sequences` raises on mixed dtype or device (#787),
`pad_*` with `pad_size=0` returns a new tensor (#788), the edge cases of the masked reductions,
`permute_*`, `lengths_to_mask` and padding are documented and pinned (#789), and
`design_principles.md` carries the closed P2 list, P10, P16 and the container-kind gate (#791).
Left open from that version: the index-select split (D2, now step 1), the stable `coola` 2.0
(step 5) and the stable-API decision (D3).
