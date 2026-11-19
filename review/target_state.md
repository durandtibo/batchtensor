# `batchtensor`: target state and the changes to reach it

**Scope.** This document defines the properties of `batchtensor` (`constants`, `utils`, `tensor`,
`nested`), compares them with `main` (`e025c10`), and lists the changes that remain to reach the
target state. It changes no code. It replaces the previous version, whose steps 1 to 6 are merged
(Appendix B). Each finding is tagged **[verified]** (a command or a probe shows it) or **[reading]**
(found by reading the code, not run).

**Where the package stands.** The architecture is at its target: `ruff`, `ty` and `lint-imports`
are clean, 2 615 tests pass (3 skipped, no accelerator), and every nested function is one call to
one of three helpers. The container walk of `combine_tensors` is now `coola`'s
(`recursive_zip_apply`), so no code of `batchtensor` traverses a container. What is left is not a
redesign. It is (1) three places where a function does something other than what a property says
(`pad_sequences` casts silently, `pad_*` with `pad_size=0` returns its input, masked reductions
leak a sentinel on an empty mask), (2) properties P2 and P8 whose exception lists are incomplete,
(3) two edge rules that are only implicit (container kinds in `cat_*`, what "permutation" checks),
and (4) a release that is blocked by an alpha dependency.

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
   not added to a function. It is documented instead (P10).

## 1. Properties of the package

These are the properties the package guarantees by construction. Each has an owner (the code that
enforces it) and a gate (the test or job that fails when it breaks). `P16` is new; `P2`, `P7`, `P8`
and `P10` change (the change is in the statement).

| #   | Property                  | Statement                                                                                                                                                                                                                                                                                                        | Owner                           | Gate                                                        |
| --- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------- |
| P1  | Fixed layout              | Batch is dim 0, sequence is dim 1. The dims come only from `BATCH_DIM` and `SEQ_DIM`. No function of `tensor` or `nested` takes a `dim`. `utils.structure` is dim-generic (the layer below).                                                                                                                     | `constants.py`                  | `test_properties`                                           |
| P2  | Thin wrappers             | A `tensor` function is one torch call with the dim fixed: same semantics, same result, no hidden copy, cast or device move. The **closed** list of exceptions is in Section 3.5 (a fixed composition, or a check torch does not make). A function not in the list is a thin wrapper.                             | `tensor/*`                      | `test_torch_reference`, **new: exception list test (T7)**   |
| P3  | Two symmetric levels      | Every `tensor` function that maps batch data has a `nested` twin with the same name, options and defaults. The exceptions are listed in the parity test.                                                                                                                                                         | Parity test                     | `test_api_parity`                                           |
| P4  | One traversal             | Every nested function goes through `apply_to_tensors`, `split_tensors` or `combine_tensors`. No nested function walks a container. `from_numpy` (array leaves) calls `recursive_apply` and is the one listed exception.                                                                                          | `utils/structure.py`            | `test_properties`                                           |
| P5  | Structure-preserving      | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple`, named tuple, `OrderedDict`, `defaultdict`. For `cat_*`, `stack_*` and `pad_sequences` the container types of the output are the ones of the first item.                                      | `apply_to_tensors`              | `test_differential`, `test_containers`                      |
| P6  | Tensor leaves only        | A leaf that is not a `torch.Tensor` raises a `TypeError` that names the leaf type. Nothing is skipped silently. Only `from_numpy` takes other leaves.                                                                                                                                                            | `apply_to_tensors`              | `test_non_tensor_leaf`                                      |
| P7  | Pure                      | Inputs are never modified. **The output never aliases the input, except where torch returns a view or the input itself (`to`, `contiguous`, `detach`, listed in Section 3.5).** A function adds no shortcut that returns its input.                                                                              | Thin wrappers                   | `test_properties`, **new: aliasing test (T8)**              |
| P8  | One error contract        | A bad leaf type is a `TypeError`. An argument that `batchtensor` validates is a `ValueError`. Anything else is the torch error, unchanged. `batchtensor` validates only what torch cannot see or what torch accepts but means something else. The table is in `design_principles.md`.                            | Validators in `utils`, `tensor` | `test_error_contract` (one row per line of the table)       |
| P9  | Explicit randomness       | Randomness comes only from `generator` (a `torch.Generator` or an `int`). Without it, the global torch RNG. One permutation is shared by all leaves.                                                                                                                                                             | `utils.seed`                    | `test_seed`, `test_device`                                  |
| P10 | Device and dtype agnostic | No function moves **data** between devices or casts a dtype. An index or permutation argument may be moved to the device of the data (it is not data). No host-device sync, except `lengths_to_mask` without `max_len`. **A function that would need a sync to validate documents the behaviour instead (P16).** | Thin wrappers                   | `test_device` (skipped without accelerator), **T9 (dtype)** |
| P11 | Differentiable            | Autograd flows through every function whose torch op is differentiable. Only `to_numpy` and `detach` cut the graph.                                                                                                                                                                                              | Thin wrappers                   | `test_autograd`                                             |
| P12 | Keyword-only options      | Positional arguments are the data and the arguments that define the operation. Options are keyword-only. No `**kwargs` forwarding, except `nested.to`. A torch option is renamed only when it shadows a builtin (`sorted` -> `sort_output`).                                                                     | Signatures                      | `test_signatures`                                           |
| P13 | Typed                     | A map function has the type `T -> T`, `T` bound to `NestedTensor`. Checked by `ty` and `pyright`.                                                                                                                                                                                                                | Annotations                     | `ty` and `pyright` in CI, `test_typing`                     |
| P14 | Layered                   | `constants` <- `utils` <- `tensor` <- `nested`. No import in the other direction. `torch` and `coola` are the only required dependencies. `numpy` is lazy.                                                                                                                                                       | `import-linter`                 | `ci-cyclic-imports`                                         |
| P15 | Documented once           | Every public function has a docstring with a runnable `pycon` example. Each rule is stated in one place (`design_principles.md`) and not repeated in the function docstrings.                                                                                                                                    | Docstrings, docs                | `ci-doctest`, `test_docstring_does_not_repeat_layout`       |
| P16 | Defined on edge cases     | **Every function has a documented result on empty and degenerate input: a structure without tensor, a zero-length dim, a fully masked row, a zero pad.** "Undefined" is not an answer; "torch's behaviour" is.                                                                                                   | Docstrings (`Note`/`Returns`)   | **New: edge-case table test (T5, T10)**                     |

**Not guaranteed:**

- Shared references. `clone({"a": x, "b": x})` returns two tensors.
- Memory sharing between input and output, except for the views listed in Section 3.5.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`shuffle_*`, `chunk_*`,
  `split_*`).
- 0-d tensors. `sum_along_batch(torch.tensor(1.))` returns `tensor(1.)` because torch accepts
  `dim=0` on a scalar. It is torch's behaviour (P8) and has no batch meaning.
- Dataclasses and sets (D1). They work today through `coola` in all five function families. No test,
  doc or type promises it.
- That an index or a `permutation` argument is a valid permutation (Section 3.4).

## 2. What differs today

Surface: 72 names in `tensor`, 81 in `nested` (80 functions and `NestedTensor`). The previous
findings are closed (Appendix B). The findings below are new.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                   | Tag        | Property |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- |
| F1  | **`pad_sequences` casts and copies across devices silently.** `pad_sequences([long tensor [1], float tensor [1.5, 2.5]])` returns `[[1, 0], [1, 2]]`: the output takes the dtype of the first tensor and `out[i] = tensor` truncates `1.5` and `2.5`. The same assignment copies a tensor of another device. `torch.cat` promotes dtypes and raises on devices; this function does neither.                                               | [verified] | P8, P10  |
| F2  | **`pad_along_*(..., pad_size=0)` returns its input object.** `pad_along_seq(x, 0) is x` is `True`, while the path with `pad_size > 0` returns a new tensor and `torch.nn.functional.pad(x, [0, 0])` also returns a new tensor. The shortcut is an extra branch that creates aliasing, and a caller that modifies the result in place modifies the input.                                                                                  | [verified] | P7       |
| F3  | **A fully masked row has no stated result.** `masked_mean_along_seq` returns `nan`, `masked_amax_along_seq` returns `torch.finfo(dtype).min` (`-3.4e38`, or `-9223372036854775808` for `int64`) and `masked_amin_along_seq` the `max`. The values are consistent but undocumented, and a sentinel flows into the next layer without a signal. A check needs a sync (constraint 7).                                                        | [verified] | P16      |
| F4  | **The exception list of P2 is incomplete.** Not one torch call: `index_select_along_seq` with a 2-D index (a per-sample `tensor[batch_index, index]` gather, while `index_select_along_batch` rejects a 2-D index), `repeat_along_*` (builds the `repeat` sizes), `pad_sequences`, `lengths_to_mask`, `shuffle_*` (permutation + `index_select`), `pad_*` (checks the value, then `F.pad`). `design_principles.md` names only `masked_*`. | [verified] | P2       |
| F5  | **The error table of P8 is incomplete.** Present in the code, absent from the table: `repeat_along_*`, `seq_len` and `nested.shuffle_*` raise `ValueError` when a tensor has too few dims, while `select_along_seq`, `slice_along_seq` and `sum_along_seq` on a 1-D tensor raise torch's `IndexError`. `lengths_to_mask` accepts a negative length (row of `False`) and a `max_len` smaller than a length (truncation) silently.          | [verified] | P8, P16  |
| F6  | **`permute_*` accepts a repeated index.** `permute_along_batch(x, tensor([0, 0, 0]))` returns three copies of row 0. It is `index_select` with a length check. The name promises a permutation, the check cannot verify it without a sync, and a wrong index silently drops samples.                                                                                                                                                      | [verified] | P8, P16  |
| F7  | **The kind of container of `cat_*` and `stack_*` items is not stated.** `cat_along_batch([[x], (x,)])` returns a list, `cat_along_batch([dict, OrderedDict])` returns a `dict`: the output follows the first item and the mismatch is accepted. The docstring of `combine_tensors` says it raises on "the same kind of containers", which is true only for mapping vs sequence.                                                           | [verified] | P5       |
| F8  | **The release is blocked by an alpha dependency and the metadata contradicts the version.** `dependencies = ["coola>=2.0.0a4,<3.0", ...]`: `pip` skips pre-releases, so users need `--pre`. The classifier says `4 - Beta` and the version is `0.1.2a1`.                                                                                                                                                                                  | [verified] | P14      |
| F9  | **Dataclasses and sets work everywhere but are out of the contract.** After #785 `sum_along_batch(D(x))` and `cat_along_batch([D(x), D(x)])` both work, so the old gap is closed. Nothing tests that they work or fail: a `coola` change could break users and no gate would notice. This is accepted (D1); it is listed so it is not mistaken for a promise.                                                                             | [verified] | P5       |
| F10 | **The cost of an operation is still four functions** (`tensor` x `batch`/`seq` x `nested`, four docstrings of about 35 lines, four test blocks). The admission rule (Section 3.6) is the answer and is merged; there is no further action.                                                                                                                                                                                                | [verified] | P15      |

**Behaviours that must not regress:** the container types of the input are kept (named tuple,
`OrderedDict`, `defaultdict`); `max`, `min`, `median`, `sort` and `topk` keep their
`torch.return_types.*` leaves; one permutation is shared by all leaves of a `shuffle_*`; `chunk_*`
and `split_*` raise on leaves of different sizes; a non-tensor leaf raises a `TypeError` that names
the type; a cycle raises `ValueError`; the doctests run.

## 3. Architecture

The architecture stays. It has three layers and one traversal module.

```text
constants          BATCH_DIM, SEQ_DIM
   ^
utils              types      NestedTensor, T
                   structure  apply_to_tensors, split_tensors, combine_tensors, get_size
                   seed       setup_torch_generator, random_permutation
   ^
tensor             one torch call with the dim fixed (+ the closed list of Section 3.5)
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

### 3.2 The container rule (D1)

The supported structure is `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). This is exactly `NestedTensor`. For the N -> 1 helpers the output
containers are the ones of the first item (F7); the other items must have the same keys and
lengths. Sets and dataclasses are outside the contract (F9).

### 3.3 One statement of each rule

- The layout (P1), the container rule, the cross-leaf rule, the empty-structure rule and the error
  table are in `design_principles.md` and nowhere else. The module docstrings of `tensor` and
  `nested` link to it.
- A function docstring states only what is specific to the function: its edge cases (P16), its
  `Raises`, and its example.

### 3.4 Edge cases: document, do not guess (P16, constraint 7)

When a check needs the values of a tensor, the function does not make it. The docstring states the
result instead, and a test pins it:

| Function                                 | Case                         | Result (documented and tested)                           |
| ---------------------------------------- | ---------------------------- | -------------------------------------------------------- |
| `masked_mean_along_seq`                  | a row without valid position | `nan` (0 / 0)                                            |
| `masked_amax_along_seq`, `masked_amin_*` | a row without valid position | the lowest (highest) value of the dtype, a sentinel      |
| `masked_sum_along_seq`                   | a row without valid position | `0`                                                      |
| `permute_*`                              | a repeated index             | not checked: it is `index_select`; the length is checked |
| `lengths_to_mask`                        | a negative length            | a row of `False`                                         |
| `lengths_to_mask`                        | `max_len` below a length     | the row is truncated to `max_len`                        |
| `pad_*`                                  | `pad_size == 0`              | a new tensor equal to the input (F2)                     |
| `pad_sequences`                          | tensors of different dtype   | `ValueError` (F1)                                        |
| `pad_sequences`                          | a zero-length tensor         | a row of `value`                                         |

### 3.5 The closed list of exceptions to P2 and P7

A function outside this list is one torch call with the dim fixed, and returns a new tensor or the
view torch returns.

| Function                                | Why it is not one torch call                                         |
| --------------------------------------- | -------------------------------------------------------------------- |
| `masked_{sum,mean,amax,amin}_along_seq` | `masked_fill` + reduction; mask validated (no sync)                  |
| `lengths_to_mask`                       | no torch equivalent; a sync only without `max_len`                   |
| `pad_sequences`                         | no torch equivalent (`torch.nn.utils.rnn.pad_sequence` is seq-first) |
| `pad_along_*`                           | `F.pad` with the dim fixed; checks `value` against the dtype         |
| `repeat_along_*`                        | builds the `repeat` sizes; checks the number of dims                 |
| `index_select_along_seq`, 2-D index     | per-sample gather (D2)                                               |
| `permute_*`, `shuffle_*`                | `index_select` + a shape check (`shuffle_*` draws the permutation)   |
| `topk_*`                                | option `sort_output` for `sorted` (builtin shadowing, P12)           |

Views (aliasing allowed): `slice_*`, `select_*`, `chunk_*`, `split_*`, `detach`, `from_numpy`,
`to_numpy` on CPU. Returns the input when nothing changes (torch's rule): `to`, `contiguous`,
`pin_memory`. Everything else returns a new tensor (T8).

### 3.6 Admission rule for a new operation

An operation enters `tensor` and `nested` only if all of these hold. It is merged in `CONTRIBUTING.md`
and `architecture.md`.

1. It needs the fixed dim (`torch.f(x, dim=BATCH_DIM)`), or it composes two torch calls in a way
   that is easy to get wrong (`masked_*`, `pad_sequences`). A pointwise function
   (`exp`, `clamp`) does not: use `apply_to_tensors(data, torch.exp)`.
2. It has a `nested` twin of shape 1 -> 1, 1 -> N or N -> 1 (Section 3.1).
3. It comes with the four functions (batch and seq, tensor and nested) or a stated reason why only
   one dim makes sense (as `masked_*_along_seq`).
4. If it is not one torch call, it is added to the list of Section 3.5.
5. It has a runnable example, a documented edge-case result (P16), and a gate for each property it
   touches.

## 4. Public API after the change

No function is added or removed unless D2 is accepted. Changes:

| Name                                                      | Change                                                                                                        | Breaking   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| `pad_sequences` (both levels)                             | Tensors of different dtype or device raise a `ValueError` (F1). The output dtype is the dtype of the tensors. | Yes        |
| `pad_along_*` (both levels)                               | `pad_size == 0` returns a new tensor (F2). No other change.                                                   | Yes (rare) |
| `masked_*_along_seq`                                      | Result on a fully masked row documented and tested (F3). No code change.                                      | No         |
| `lengths_to_mask`, `permute_*`                            | Edge cases documented and tested (F5, F6). No code change.                                                    | No         |
| `combine_tensors` and `cat_*`, `stack_*`, `pad_sequences` | Docstrings say the container types of the output are the ones of the first item (F7).                         | No         |
| `index_select_along_seq`                                  | D2: 2-D index moves to its own function, or stays and is listed in Section 3.5.                               | Yes if D2  |
| `design_principles.md`                                    | P2 list (Section 3.5), P7, P8 table rows (F5), P10 wording and P16 added.                                     | No         |
| `pyproject.toml`                                          | `coola>=2.0,<3.0` when stable (F8); classifier and version set together.                                      | No         |

## 5. Dependents

| Package             | Change                                                                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola`             | Done: `recursive_zip_apply` (`2.0.0a4`) and the `rebuild_*` exports (`a3`). Left: a stable `2.0` release (F8). No other request.                                  |
| `batchtensor` users | Code that passes `pad_sequences` tensors of mixed dtype now gets a `ValueError` (cast them first). Code that relied on `pad_along_*(x, 0) is x` is not supported. |

## 6. Evidence

### 6.1 Cost (probes on `main`, Python 3.14, CPU, 200 keys x 5 tensors of shape `(4, 3)` = 1 000 leaves)

| Call                                              | Time per call |
| ------------------------------------------------- | ------------- |
| `nested.sum_along_batch(data)`                    | 2.3 ms        |
| hand-written dict/list comprehension, same result | 1.1 ms        |
| `nested.chunk_along_batch(data, 2)`               | 2.6 ms        |
| `nested.cat_along_batch([data, data])`            | 2.1 ms        |

The overhead is about 1.2 microseconds per leaf (about 2x the comprehension on tiny tensors) and
negligible on real batches. Constraint 5 holds. The numbers are one run, not a benchmark.

### 6.2 Probes of behaviour

| Probe                                                                | Result                                                        |
| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| `pad_sequences([long [1], float [1.5, 2.5]])`                        | `[[1, 0], [1, 2]]`, silent truncation (F1)                    |
| `pad_along_seq(x, 0) is x`                                           | `True` (F2)                                                   |
| `masked_mean_along_seq` with a fully masked row                      | `nan` (F3)                                                    |
| `masked_amax_along_seq` on `float32` / `int64`, fully masked row     | `-3.4028e+38` / `-9223372036854775808` (F3)                   |
| `permute_along_batch(x, tensor([0, 0, 0]))`                          | three copies of row 0 (F6)                                    |
| `permute_along_batch(x, tensor([0, 1, 7]))`                          | torch's `IndexError` (P8)                                     |
| `lengths_to_mask(tensor([-1, 2]))`                                   | `[[False, False], [True, True]]` (F5)                         |
| `lengths_to_mask(tensor([3, 2]), max_len=2)`                         | truncated to 2 columns (F5)                                   |
| `slice_along_seq(torch.ones(3), 0, 1)`                               | torch's `IndexError`; `repeat_along_seq` gives a `ValueError` |
| `cat_along_batch([[x], (x,)])`, `[dict, OrderedDict]`                | list, dict (first item wins) (F7)                             |
| `cat_along_batch([{"a": x}, {"b": x}])`                              | `ValueError`: keys differ                                     |
| `cat_along_batch([])`                                                | `ValueError`                                                  |
| `sum_along_batch(D(x))`, `cat_along_batch([D(x), D(x)])` (dataclass) | both work (F9)                                                |
| `sum_along_batch({"a": "abc"})`                                      | `TypeError` naming `str`                                      |
| `sum_along_batch` on a dict that contains itself                     | `ValueError: Cycle detected`                                  |
| `shuffle_along_seq` on leaves of seq 4 and 5                         | `ValueError`: sizes `[4, 5]`                                  |
| `shuffle_along_batch` on two leaves, one seed                        | one permutation for both                                      |
| `from_numpy({"a": tensor})`, `from_numpy({"a": [1, 2]})`             | `TypeError` of `torch.from_numpy`                             |
| `mean_along_batch(torch.arange(3))`                                  | torch's `NotImplementedError`, unchanged (P8)                 |
| `topk_along_batch(x, 5)` with 3 rows                                 | torch's `RuntimeError`, unchanged                             |

Commands: `ruff check src tests` clean, `ty check` clean, `lint-imports` 1 contract kept,
`pytest tests` 2 615 passed, 3 skipped. No accelerator was available, so the device gates did not
run.

## 7. Changes to reach the target state

One pull request per step, each releasable. Steps 1 to 4 are independent.

| Step | Change                                                                                                                                                                                                                              | Breaking   | Status                         |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------ |
| 1    | `pad_sequences`: raise a `ValueError` on tensors of different dtype or device (F1). Regression tests fail without the fix. Update the docstring, the migration guide and the error table.                                           | Yes        | PR #787                        |
| 2    | `pad_*`: delete the `pad_size == 0` shortcut so the result is a new tensor (F2). Aliasing test T8 for all functions (list of views in Section 3.5).                                                                                 | Yes (rare) | PR #788                        |
| 3    | Document and pin the edge cases of Section 3.4 (masked reductions, `permute_*`, `lengths_to_mask`, `pad_*`, `pad_sequences`). One table test T10 with the fully specified outputs. No code change.                                  | No         | PR #789                        |
| 4    | Update `design_principles.md`: the closed list of P2 (Section 3.5), P7, the rows of P8 (F5), P10, P16; fix the "same kind of containers" sentence in `combine_tensors` and state the first-item rule (F7). Add the tests T7 and T9. | No         | PR #791                        |
| 5    | Decide D2 (`index_select_along_seq` 2-D). If accepted: add `gather_along_seq` (both levels), restrict `index_select_along_seq` to a 1-D index, and move the tests.                                                                  | Yes if D2  | Blocked on D2                  |
| 6    | When `coola` 2.0 is stable: depend on `coola>=2.0,<3.0` (F8).                                                                                                                                                                       | No         | Blocked: no stable `coola` 2.0 |
| 7    | Set the classifier `Development Status` and the version to the first non-alpha release. Remove "the API is not stable" from `design_principles.md` only when D3 is answered.                                                        | No         | Blocked: step 6 and D3         |

Steps 1 to 4 can ship now. Step 6 blocks the release.

## 8. Tests and gates

- **Kept:** properties (`test_properties`), parity and signatures, differential, torch reference,
  autograd, device, non-tensor leaf, error contract (`test_error_contract`), containers, typing
  (`ty` and `pyright`), doctests. 100% coverage of the code added or changed. A bug fix adds a
  regression test that fails without the fix (`git stash push src`).
- **T5. Empty structure** (merged as part of the contract): `{}`, `[]`, `()` are kept by maps and
  raise `ValueError` in `chunk_*`, `split_*`, `shuffle_*`.
- **T7. Exception list of P2 (new).** A test reads the `tensor` modules and checks that each
  function body is one `return` of one torch call, or that the function is in the list of
  Section 3.5. A new function that is not a thin wrapper fails until it is listed.
- **T8. Aliasing (new).** For each function of the `tensor` API, the output shares no memory with
  the input unless the function is in the list of views or of "returns the input" (Section 3.5).
  Gate for F2.
- **T9. No cast (new).** For each function, the dtype of the output is the dtype torch gives for
  the same call. Gate for F1 and P10. `pad_sequences` with mixed dtypes raises.
- **T10. Edge-case table (new).** One parametrized test over Section 3.4 with the output asserted
  against a literal (fully masked row, repeated index, negative length, zero pad, zero-length
  tensor).
- **T11. Container kind (new).** `cat_*` and `stack_*` of `[list, tuple]` and `[dict, OrderedDict]`
  return the type of the first item; mapping vs sequence raises `ValueError`. Gate for F7.
- **No test** says that sets or dataclasses are supported (D1).

## 9. Decisions for the maintainer

| #   | Decision                                                                | Recommendation                                                                                                                                                                                                                                                        |
| --- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Dataclasses and sets: supported or outside the contract?                | **Outside the contract.** They work through `coola` and a user can rely on it at their own risk. A promise needs a type (`NestedTensor` cannot express a dataclass) and a gate in five families, for a use case where batches are dicts.                              |
| D2  | `index_select_along_seq` with a 2-D index: keep or split?               | **Split.** A 2-D index is a per-sample gather, not `index_select`: `index_select_along_batch` already rejects it, the name misleads and P2 needs an exception. A `gather_along_seq` is its own function with its own signature. Alternative: keep and list it in 3.5. |
| D3  | When is the API stable?                                                 | After step 6 and one release with no change. Until then keep "pre-1.0".                                                                                                                                                                                               |
| D4  | Fully masked rows: raise, return a sentinel, or return `nan`?           | **Document the current result (Section 3.4).** A check needs a sync (constraint 7). A user who needs a guard tests `mask.any(dim=1)`. Alternative: an option `empty_value=` on the amax/amin; it adds an option to four functions for a case the caller can handle.   |
| D5  | `pad_sequences` with mixed dtype: raise or promote like `torch.cat`?    | **Raise.** One rule (P10: no cast) and no promotion table to maintain. The caller casts in one call: `nested.to(sample, dtype=...)`.                                                                                                                                  |
| D6  | `permute_*`: validate that the index is a permutation?                  | **No.** It needs a sort or a scatter and a sync. Keep the length check and document F6. `shuffle_*` always creates a valid permutation.                                                                                                                               |
| D7  | Generate the nested twins instead of writing them?                      | **No.** Generation hides the signature from the type checkers and the docs, and a nested function is already one line. The cost is docstrings, which are the product.                                                                                                 |
| D8  | Add pointwise nested functions back (`exp`, `clamp`)?                   | **No.** `apply_to_tensors(data, torch.exp)` is the API (migration guide).                                                                                                                                                                                             |
| D9  | Drop the `tensor` level from the public docs and keep only `nested`?    | **No.** `tensor` is what people use on a single tensor, and `nested` is defined by it (P2, P3).                                                                                                                                                                       |
| D10 | Make `cat_*` strict on the kind of container (all items the same type)? | **No.** It needs a check of `batchtensor` on top of `coola`'s. Document the first-item rule (F7) and pin it with T11.                                                                                                                                                 |

## 10. Not covered

- New operations. The admission rule (Section 3.6) is the process; no operation is proposed here,
  except the split of D2.
- Ragged or jagged data (`torch.nested`), sparse tensors, `torch.compile` support.
- A `dim` argument or a configurable layout (P1 says no).
- Data loaders, datasets or collate functions (`pad_sequences` and `cat_*` are the building
  blocks, not a framework).
- Python and `torch` version policy beyond `torch>=2.4,<3.0` and Python 3.11 to 3.15.
- Performance work. Section 6.1 shows no need.
- Behaviour on an accelerator. The device gates were not run (no accelerator on the review
  machine); a CUDA run of `test_device` is required before step 7.

## Appendix A. API inventory

**`tensor` (72 names).** Reductions (`amax`, `amin`, `argmax`, `argmin`, `max`, `mean`, `median`,
`min`, `prod`, `sum`, `std`, `var`, `any`, `all`, `logsumexp`), cumulative (`cumsum`, `cumprod`),
sorting (`sort`, `argsort`, `topk`), indexing (`select`, `index_select`), slicing (`slice`,
`chunk`, `split`), joining (`cat`, `stack`, `repeat`), permutation (`permute`, `shuffle`, `flip`,
`roll`), padding (`pad`), each `_along_batch` and `_along_seq`; plus
`masked_{sum,mean,amax,amin}_along_seq`, `lengths_to_mask`, `pad_sequences`.

**`nested` (80 functions).** The same names except `lengths_to_mask`, plus `clone`, `detach`,
`contiguous`, `pin_memory`, `to`, `to_numpy`, `from_numpy`, `batch_size`, `seq_len`, and the type
`NestedTensor`.

**`utils`.** `structure` (`apply_to_tensors`, `split_tensors`, `combine_tensors`, `get_size`),
`seed` (`setup_torch_generator`, `random_permutation`), `types` (`NestedTensor`, `T`).

**`constants`.** `BATCH_DIM`, `SEQ_DIM`.

## Appendix B. Record of the previous target states

The first version listed 11 steps, all merged: the three structure helpers, keyword-only options,
removal of the pointwise nested functions, the validation contract, `nested.batch_size` and
`seq_len`, `pad_sequences`, the masked reductions, `pyright` in CI, the gates for P1, P4 and P7,
and the migration guide.

The second version (steps 1 to 6 merged): `architecture.md` rewritten from the properties (#777),
the layout `Note:` blocks removed (#778), the container, cross-leaf and empty-structure rules
(#779), the error contract test (#780), the admission rule (#781), `coola>=2.0.0a3` and the
`rebuild_*` import from `coola.recursive` (#782), and `combine_tensors` on
`coola.recursive.recursive_zip_apply` (#785), which closed the old F6 (private container walk) and
the old F2 (dataclasses in `cat_*`). Open from that version: the stable `coola` 2.0 (now F8, step 6)
and the stable-API decision (D3).
