# `batchtensor`: target state and the changes to reach it

**Scope.** This document defines the properties of `batchtensor` (`constants`, `utils`, `tensor`,
`nested`), compares them with `main` (`edcf6cd`), and lists the changes that remain to reach the
target state. It changes no code. It replaces the previous version of this file, which proposed an
11-step refactoring that is now merged (see Appendix B). Each finding is tagged **[verified]** (a
command or a probe shows it) or **[reading]** (found by reading the code, not run).

**Where the package stands.** The design is close to the target: `ruff`, `ty` and `lint-imports`
are clean, `pyright` runs in CI, and 2 406 tests pass (3 skipped, no accelerator). Every nested
function is one call to one of three helpers. What is left is not a redesign. It is (1) a
dependency on a pre-release of `coola` that blocks a stable release, (2) one container rule that
the code and the docs state differently (dataclasses), (3) stale documentation that contradicts
the properties, (4) the cost of the four-fold repetition of every operation, and (5) the rule that
keeps the surface from growing without a reason.

This is a pre-1.0 release (`0.1.2a1`). Breaking changes are allowed when they make the package
simpler.

## Constraints

1. **Simple to maintain.** A rule is implemented in one place. A function is a thin wrapper. Remove
   code instead of adding a workaround.
2. **Breaking changes are allowed.** No deprecation period, no adapter. The future state matters
   more than the past state.
3. **Consistent with `coola`.** Container handling (types, keys, order) is the one of
   `coola.recursive`. `batchtensor` does not re-implement it, except where `coola` has no
   multi-input traversal (`combine_tensors`).
4. **Follow `CLAUDE.md`.** Batch is dim 0, sequence is dim 1, `tensor` and `nested` stay in sync,
   options are keyword-only, shared helpers are public functions of `batchtensor.utils`, optional
   libraries are imported lazily.
5. **No slower than today.** A nested call is dominated by the torch call (Section 6.1).
6. **A claim has a gate.** A property written in the docs is enforced by a test or a CI job that
   fails when it breaks. A property without a gate is given one or removed.

## 1. Properties of the package

These are the properties the package guarantees by construction. Each has an owner (the code that
enforces it) and a gate (the test or job that fails when it breaks).

| #   | Property                  | Statement                                                                                                                                                                                                       | Owner                              | Gate                                          |
| --- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | --------------------------------------------- |
| P1  | Fixed layout              | Batch is dim 0, sequence is dim 1. The dims come only from `BATCH_DIM` and `SEQ_DIM`. No function of `tensor` or `nested` takes a `dim`. `utils.structure` is dim-generic (the layer below).                    | `constants.py`                     | `test_properties`                             |
| P2  | Thin wrappers             | A `tensor` function is one torch call with the dim fixed. Same semantics and result as torch, no hidden copy, cast or device move. The one exception is `masked_*_along_seq`, a fixed `masked_fill` + reduction. | `tensor/*`                         | `test_torch_reference`                        |
| P3  | Two symmetric levels      | Every `tensor` function that maps batch data has a `nested` twin with the same name, options and defaults. The exceptions are listed in the parity test.                                                         | Parity test                        | `test_api_parity`                             |
| P4  | One traversal             | Every nested function goes through `apply_to_tensors`, `split_tensors` or `combine_tensors`. No nested function walks a container.                                                                              | `utils/structure.py`               | `test_properties`                             |
| P5  | Structure-preserving      | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple`, named tuple, `OrderedDict`, `defaultdict` (`coola.recursive` rules). Dataclasses: see D1.   | `apply_to_tensors`                 | `test_differential`, `test_structure`         |
| P6  | Tensor leaves only        | A leaf that is not a `torch.Tensor` raises a `TypeError` that names the leaf type. Nothing is skipped silently. Only `from_numpy` takes other leaves.                                                           | `apply_to_tensors`                 | `test_non_tensor_leaf`                        |
| P7  | Pure                      | Inputs are never modified. A function returns a view where torch returns a view (documented), otherwise a new tensor.                                                                                           | Thin wrappers                      | `test_properties`                             |
| P8  | One error contract        | A bad leaf type is a `TypeError`. An argument that `batchtensor` validates is a `ValueError`. Anything else is the torch error, unchanged. `batchtensor` validates only what torch cannot see.                  | Validators in `utils`, `tensor`    | Unit tests (no table test, see T3)            |
| P9  | Explicit randomness       | Randomness comes only from `generator` (a `torch.Generator` or an `int`). Without it, the global torch RNG. One permutation is shared by all leaves.                                                            | `utils.seed`                       | `test_seed`, `test_device`                    |
| P10 | Device and dtype agnostic | No function moves data between devices or casts a dtype. No host-device sync except `lengths_to_mask` without `max_len`.                                                                                        | Thin wrappers                      | `test_device` (skipped without accelerator)   |
| P11 | Differentiable            | Autograd flows through every function whose torch op is differentiable. Only `to_numpy` and `detach` cut the graph.                                                                                             | Thin wrappers                      | `test_autograd`                               |
| P12 | Keyword-only options      | Positional arguments are the data and the arguments that define the operation. Options are keyword-only. No `**kwargs` forwarding, except `nested.to`.                                                          | Signatures                         | `test_signatures`                             |
| P13 | Typed                     | A map function has the type `T -> T`, `T` bound to `NestedTensor`. Checked by `ty` and `pyright`.                                                                                                               | Annotations                        | `ty` and `pyright` in CI, `test_typing`       |
| P14 | Layered                   | `constants` <- `utils` <- `tensor` <- `nested`. No import in the other direction. `torch` and `coola` are the only required dependencies. `numpy` is lazy.                                                       | `import-linter`                    | `ci-cyclic-imports`                           |
| P15 | Documented once           | Every public function has a docstring with a runnable `pycon` example. Each property is stated in one place (`design_principles.md`) and nowhere else.                                                          | Docstrings, docs                   | `ci-doctest`; **no gate for "stated once"**   |

**Not guaranteed:**

- Shared references. `clone({"a": x, "b": x})` returns two tensors.
- Memory sharing between input and output, except for the documented views.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`shuffle_*`, `chunk_*`,
  `split_*`).
- 0-d tensors. `sum_along_batch(torch.tensor(1.))` returns `tensor(1.)` because torch accepts
  `dim=0` on a scalar. It is torch's behaviour (P8) and has no batch meaning.

## 2. What differs today

Surface: 72 names in `tensor`, 81 in `nested` (80 functions and `NestedTensor`). `src` has 7.5k
lines, of which 5.7k are docstrings.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                    | Tag        | Property |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- |
| F1  | **The runtime dependency is a pre-release, and the code still uses the old import path.** `dependencies = ["coola>=2.0.0a2,<3.0", ...]`. A stable `batchtensor` cannot depend on an alpha: `pip` skips pre-releases, so users need `--pre`. `coola 2.0.0a3` now exports `rebuild_mapping` and `rebuild_sequence` from `coola.recursive`, but `utils/structure.py` still imports them from `coola.recursive.rebuild`, which works on `a2` only. | [verified] | P4, P14  |
| F2  | **Dataclasses are claimed and half supported.** Map functions, `chunk_*`, `split_*` and `shuffle_*` accept a dataclass at runtime. `cat_*`, `stack_*` and `pad_sequences` raise `TypeError`. Both type checkers reject a dataclass argument because `NestedTensor` excludes it. A claim that is true for 3 of 5 function families, and untyped for all, is not a property.                                  | [verified] | P5, P13  |
| F3  | **`docs/docs/architecture.md` contradicts the properties.** It teaches `recursive_apply` as the nested pattern (Pattern 2, "Adding New Nested Operations", step 2), says functions are named `operation_along_dimension`, says tensor functions "return a new tensor" (views exist), shows `keepdim` as positional, and lists "Future Directions" that are not a plan. `design_principles.md` is right.      | [verified] | P4, P15  |
| F4  | **136 identical `Note:` blocks.** Every function carries "This function assumes the batch dimension is the first dimension of the tensors. All the tensors should have the same batch size." (about 4 lines, 136 times, about 550 lines). It restates P1 on every function, and for nested functions the "same size" sentence is the cross-leaf rule that P8 says is *not* validated by map functions.      | [verified] | P1, P15  |
| F5  | **The cost of an operation is four functions.** One operation is `tensor` x `batch`/`seq` x `nested` = 4 functions, 4 docstrings of about 35 lines, 4 test blocks. 77% of `tensor` and `nested` lines are docstrings. The code is thin; the repetition is the maintenance cost, and nothing states when an operation is worth that cost.                                                                    | [verified] | P15      |
| F6  | **`combine_tensors` has its own container walk.** `coola` has no multi-input recursion, so `_combine_tensors` re-implements the `Mapping` and `list`/`tuple` cases. It differs from `recursive_apply` on sets and dataclasses (F2). It is covered by `test_differential` for the supported types, but nothing else keeps the two walks equal.                                                              | [reading]  | P4, P5   |
| F7  | **`split_tensors` traverses three times.** `apply_to_tensors` wraps each tuple in `_Parts`, `dfs_iterate` counts the pieces, and `recursive_apply` runs once per piece. Result: `chunk_along_batch` on 1 000 tensors costs 4.2 ms against 1.3 ms for `sum_along_batch` (verified, Section 6.1). The cost is small and simple. It stays unless a profile asks for more.                                    | [verified] | -        |
| F8  | **Empty structures are not consistent, by rule.** `sum_along_batch({})` returns `{}`; `chunk_along_batch({}, 2)` and `shuffle_along_batch({})` raise `ValueError`. The rule (a map keeps the empty structure, an operation that needs a size raises) is right, but it is stated nowhere.                                                                                                                  | [verified] | P8       |
| F9  | **P8 has no table test.** The error contract is in `design_principles.md` as a table, tested function by function. A new function can break the contract and no test names the contract.                                                                                                                                                                                                                  | [reading]  | P8       |
| F10 | **Two `NestedTensor` definitions of "supported" exist.** The alias allows `Mapping`, `list`, `tuple`. The runtime allows anything `coola` rebuilds (sets, dataclasses). The gap is F2 and sets.                                                                                                                                                                                                              | [verified] | P13      |

**Behaviours that must not regress:** the container types of the input are kept (named tuple,
`OrderedDict`, `defaultdict`); `max`, `min`, `median`, `sort` and `topk` keep their
`torch.return_types.*` leaves; one permutation is shared by all leaves of a `shuffle_*`; `chunk_*`
and `split_*` raise on leaves of different sizes; the doctests run.

## 3. Architecture

The architecture stays. It has three layers and one traversal module.

```text
constants          BATCH_DIM, SEQ_DIM
   ^
utils              types      NestedTensor, T
                   structure  apply_to_tensors, split_tensors, combine_tensors, get_size
                   seed       setup_torch_generator, random_permutation
   ^
tensor             one torch call with the dim fixed (+ masked_*, lengths_to_mask, pad_sequences)
   ^
nested             one helper call per function: partial(bt.f, **options)
```

### 3.1 A nested function is logic-free

```python
def sum_along_batch(data: T, *, keepdim: bool = False) -> T:
    return apply_to_tensors(data, partial(bt.sum_along_batch, keepdim=keepdim))
```

Three shapes exist, and only three. A new operation uses one of them:

| Shape         | Helper                          | Used by                                               |
| ------------- | ------------------------------- | ----------------------------------------------------- |
| 1 -> 1        | `apply_to_tensors(data, fn)`    | reductions, math, indexing, slicing, padding, masking |
| 1 -> N        | `split_tensors(data, fn)`       | `chunk_*`, `split_*`                                  |
| N -> 1        | `combine_tensors(items, fn)`    | `cat_*`, `stack_*`, `pad_sequences`                   |

A function that does not fit one of these shapes is not a nested wrapper and does not belong in
`nested` (it belongs in `utils.structure` as a helper, like `get_size`).

### 3.2 The container rule (D1)

The supported structure is `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). This is exactly `NestedTensor`. The three helpers share this rule,
and `NestedTensor` is its type. Anything else (sets, dataclasses) is outside the contract: it may
work through `coola`, and no test, doc or type promises it. This removes F2 and F10 without
removing code: `apply_to_tensors` still calls `recursive_apply`.

### 3.3 One statement of each rule

- The layout (P1) is stated in `design_principles.md` and in the module docstrings of `tensor` and
  `nested`. It is not repeated in the 136 function docstrings (F4).
- The cross-leaf rule is stated once per family: map functions do not check sizes; `chunk_*`,
  `split_*`, `shuffle_*` do; `cat_*`/`stack_*` check layout.
- The empty-structure rule is stated in `design_principles.md` (F8).

### 3.4 Admission rule for a new operation (F5)

An operation enters `tensor` and `nested` only if all of these hold:

1. It needs the fixed dim (`torch.f(x, dim=BATCH_DIM)`), or it composes two torch calls in a way
   that is easy to get wrong (`masked_*`, `pad_sequences`). A pointwise function
   (`exp`, `clamp`) does not: use `apply_to_tensors(data, torch.exp)`.
2. It has a `nested` twin of shape 1 -> 1, 1 -> N or N -> 1 (Section 3.1).
3. It comes with the four functions (batch and seq, tensor and nested) or a stated reason why only
   one dim makes sense (as `masked_*_along_seq`).
4. It has a runnable example and a gate for each property it touches.

## 4. Public API after the change

No function is added or removed. The API is the one of Appendix A, with these changes:

| Name                               | Change                                                                                                    | Breaking |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- | -------- |
| Docstrings of `tensor` and `nested` | The `Note:` block is removed (F4). The batch and sequence statement moves to the module docstring.        | No       |
| `P5` and `NestedTensor`            | Dataclasses and sets leave the contract (D1). They are neither tested nor documented as supported.        | Yes (doc) |
| `docs/docs/architecture.md`        | Rewritten from the properties, 80 lines or less, no code pattern that differs from `src`.                 | No       |
| `pyproject.toml`                   | `coola>=2.0.0a3` now (F1), a stable `>=2.0` release before the first non-alpha `batchtensor`.             | No       |

## 5. Dependents

| Package      | Change                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola`      | Done in `2.0.0a3`: `rebuild_mapping`, `rebuild_sequence` and `rebuild_set` are exported from `coola.recursive`. Left: release a stable 2.0, and optionally add a multi-input traversal (F6).    |
| `batchtensor` users | Code that passes a dataclass or a set keeps working until a `coola` change breaks it. The migration guide says it is unsupported.                 |

## 6. Evidence

### 6.1 Cost (probes on `main`, Python 3.14, CPU, 200 keys x 5 tensors of shape `(4, 3)`)

| Call                                              | Time per call |
| ------------------------------------------------- | ------------- |
| `nested.sum_along_batch(data)`                    | 1.8 ms        |
| hand-written dict/list comprehension, same result | 1.3 ms        |
| `nested.chunk_along_batch(data, 2)`               | 4.2 ms        |
| `nested.cat_along_batch([data, data])`            | 1.7 ms        |

The nested overhead is below 50% on 1 000 tiny tensors and negligible on real batches. F7 is
therefore not a target.

### 6.2 Probes of behaviour

| Probe                                                         | Result                                                              |
| ------------------------------------------------------------- | ------------------------------------------------------------------- |
| `cat_along_batch` of two `defaultdict`, two `OrderedDict`     | Types kept                                                          |
| `cat_along_batch([D(x), D(x)])` (dataclass)                   | `TypeError: dataclasses are not supported by combine_tensors`       |
| `chunk_along_batch(D(x), 2)` (dataclass)                      | Works                                                               |
| `sum_along_batch({"a": x, "b": None})`, `{"a": "abc"}`        | `TypeError` naming `NoneType`, `str`                                |
| `sum_along_batch({})`                                         | `{}`                                                                |
| `chunk_along_batch({}, 2)`                                    | `ValueError: Cannot split data that does not contain any tensor`    |
| `batch_size({"a": (4, 3), "b": (5,)})`                        | `ValueError`: sizes `[4, 5]`                                        |
| `seq_len({"a": torch.ones(4)})`                               | `ValueError`: needs 2 dimensions                                    |
| `shuffle_along_batch` on leaves of size 4 and 5               | `ValueError`: sizes `[4, 5]`                                        |
| `sum_along_batch(torch.tensor(1.))`                           | `tensor(1.)` (torch's behaviour)                                    |
| `mean_along_batch(torch.arange(3))`                           | torch's `NotImplementedError`, unchanged (P8)                       |

Commands: `ruff check` clean, `ty check` clean, `lint-imports` 1 contract kept, `pytest tests`
2 406 passed, 3 skipped.

## 7. Changes to reach the target state

One pull request per step, each releasable.

| Step | Change                                                                                                                                                              | Breaking | Status |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------ |
| 1    | Rewrite `architecture.md` from Section 3 and delete the contradicting patterns (F3). Link, do not copy, the properties.                                             | No       | Done (#777) |
| 2    | Remove the 136 `Note:` blocks. Put the layout sentence in the module docstrings of `tensor` and `nested` (F4). Doctests unchanged.                                  | No       | Done (#778) |
| 3    | State the container rule (D1), the cross-leaf rule and the empty-structure rule (F8) in `design_principles.md`. Remove "dataclass" from P5, the docs and the tests that promise it; add one test that pins the documented `TypeError` of `combine_tensors` on a dataclass. | Yes (doc) | Done (#779) |
| 4    | Add the contract test T3 (P8 table).                                                                                                                                | No       | Done (#780) |
| 5    | Add the admission rule (Section 3.4) to `CONTRIBUTING.md`.                                                                                                          | No       | Done (#781) |
| 6    | Require `coola>=2.0.0a3` and import `rebuild_mapping` and `rebuild_sequence` from `coola.recursive` (F1). Verified on `a3`: tests, `ty`, `pyright` and `lint-imports` pass; `ruff` only asks to re-sort the import. Update `uv.lock`. | No       | Done (#782) |
| 7    | When `coola` 2.0 is stable: depend on `coola>=2.0,<3.0`.                                                                                                            | No       | Blocked: no stable `coola` 2.0 |
| 8    | Set the classifier `Development Status` and the version to the first non-alpha release. Remove the `README` sentence "the API is not stable" only when D3 is answered. | No       | Blocked: step 7 and D3 |

Steps 1 to 6 can ship now. Step 7 blocks the release.

## 8. Tests and gates

- **T1. Properties** (`test_properties`): P1, P4, P7. Kept.
- **T2. Parity** (`test_api_parity`) and **signatures** (`test_signatures`): kept. Add the check that
  no public function of `tensor` or `nested` has a docstring `Note:` that mentions the layout
  (keeps F4 fixed).
- **T3. Error contract (new).** One parametrized integration test over a table `(function, bad
  input, exception, match)` for the rows of the P8 table in `design_principles.md`. A row of the
  table without a test fails the review, not the CI.
- **T4. Container rule (new).** One parametrized test: the three helpers and one function of each
  family accept the same list of structures (`dict`, `list`, `tuple`, named tuple, `OrderedDict`,
  `defaultdict`, nested mix) and return the same container types. This is the gate for F6.
- **T5. Empty structure (new).** `{}`, `[]`, `()`: maps return the empty structure; `chunk_*`,
  `split_*`, `shuffle_*` raise `ValueError`.
- **T6. Pin of the documented limits.** `cat_*` on a dataclass raises `TypeError` with the message
  of `combine_tensors`.
- Keep: differential, torch reference, autograd, device, non-tensor leaf, typing (`ty` and
  `pyright`), doctests, 100% coverage of the code added or changed. A bug fix adds a regression
  test that fails without the fix.

## 9. Decisions for the maintainer

| #   | Decision                                                                                | Recommendation                                                                                                                                         |
| --- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1  | Dataclasses and sets: supported or outside the contract?                                | **Outside the contract.** `combine_tensors` cannot rebuild a dataclass, and typing them needs a protocol that `NestedTensor` cannot express. Alternative: support them everywhere, which needs `coola` to rebuild dataclasses from several items. Not worth it for the use case (batches are dicts). |
| D2  | Remove the `Note:` blocks or keep them as a reminder?                                   | **Remove.** The rule is in the module docstring and the properties. 550 lines, 136 places to update if the rule changes.                                |
| D3  | When is the API stable?                                                                 | After step 7 and one release with no change. Until then keep "pre-1.0".                                                                                |
| D4  | Generate the nested twins instead of writing them?                                      | **No.** Generation hides the signature from the type checkers and the docs, and a nested function is already one line. The cost is docstrings, which are the product. |
| D5  | Add pointwise nested functions back (`exp`, `clamp`)?                                   | **No.** `apply_to_tensors(data, torch.exp)` is the API; they were removed for that reason (migration guide).                                            |
| D6  | Add a multi-input traversal to `coola` (F6)?                                            | **Yes, if `coola` agrees.** It removes `_combine_tensors` and its container walk. Otherwise keep the current walk with T4.                              |
| D7  | Drop the `tensor` level from the public docs and keep only `nested`?                    | **No.** `tensor` is the one that people use on a single tensor, and `nested` is defined by it (P2, P3).                                                  |

## 10. Not covered

- New operations. The admission rule (Section 3.4) is the process; no operation is proposed here.
- Ragged or jagged data (`torch.nested`), sparse tensors, `torch.compile` support.
- A `dim` argument or a configurable layout (P1 says no).
- Data loaders, datasets or collate functions (`pad_sequences` and `cat_*` are the building
  blocks, not a framework).
- Python and `torch` version policy beyond `torch>=2.4,<3.0` and Python 3.11 to 3.15.
- Performance work on `split_tensors` (F7).

## Appendix A. API inventory

**`tensor` (72 names).** Reductions (`amax`, `amin`, `argmax`, `argmin`, `max`, `mean`, `median`,
`min`, `prod`, `sum`, `std`, `var`, `any`, `all`, `logsumexp`), cumulative (`cumsum`, `cumprod`),
sorting (`sort`, `argsort`, `topk`), indexing (`select`, `index_select`), slicing (`slice`,
`chunk`, `split`), joining (`cat`, `stack`, `repeat`), permutation (`permute`, `shuffle`, `flip`,
`roll`), padding (`pad`), each `_along_batch` and `_along_seq`; plus `masked_{sum,mean,amax,amin}_along_seq`,
`lengths_to_mask`, `pad_sequences`.

**`nested` (80 functions).** The same names except `lengths_to_mask`, plus `clone`, `detach`,
`contiguous`, `pin_memory`, `to`, `to_numpy`, `from_numpy`, `batch_size`, `seq_len`, and the type
`NestedTensor`.

**`utils`.** `structure` (`apply_to_tensors`, `split_tensors`, `combine_tensors`, `get_size`),
`seed` (`setup_torch_generator`, `random_permutation`), `types` (`NestedTensor`, `T`).

**`constants`.** `BATCH_DIM`, `SEQ_DIM`.

## Appendix B. Record of the previous target state

The previous version listed 11 steps, all merged: the three structure helpers
(`apply_to_tensors`, `split_tensors`, `combine_tensors`), keyword-only options, removal of the
pointwise nested functions, the validation contract, `nested.batch_size` and `seq_len`,
`pad_sequences`, the masked reductions, `pyright` in CI, the gates for P1, P4 and P7, and the
migration guide. The findings G1 to G8 of that version are fixed on `main`. Of G9 to G11, G9 (the
cost of repetition) is F5 here, G10 (small edges) is F8, and G11 (private `rebuild` import) is F1:
`coola 2.0.0a3` exports the functions, the import in `batchtensor` is step 6.
