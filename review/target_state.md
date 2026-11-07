# `batchtensor`: target state and the changes to reach it

**Scope.** This document defines the properties of `batchtensor` (`constants`, `utils`, `tensor`,
`nested`), compares them with `main`, and lists the changes that remain to reach the target state.
It changes no code. It is based on a read of the sources, the docs, the tests and the CI
configuration, and on probes run against `main` (`1009e2f`) with `coola 2.0.0a2` and the `torch`
of `.venv` (Python 3.14). Section 6 lists the probes. Each finding is tagged **[verified]** (a probe
or a command shows it) or **[reading]** (found by reading the code, not run).

**Where the package stands.** The first version of this document proposed an 11-step refactoring
(three helpers, keyword-only options, no pointwise functions, validation contract, new
operations, ...). All 11 steps are merged (Appendix B). The code is close to the first target:
1 812 tests pass, `ruff`, `ty`, `pyright` and `lint-imports` are clean. This version is therefore
not a rewrite of the design. It (1) restates the properties, (2) checks each one against `main`
and (3) lists what is left: gaps between the claims and the code, gaps between the claims and the
gates, and the next features.

This is a pre-1.0 release (`0.1.2a1`). The README says the API is not stable. The target remains a
breaking release when a breaking change makes the package simpler.

## Constraints

1. **Simple to maintain.** Every function is a thin wrapper. A rule is implemented in one place.
   Remove code instead of adding workarounds.
2. **Breaking changes are allowed.** No deprecation period and no adapters. The future state matters
   more than the past state.
3. **Consistent with `coola`.** Container handling (types, keys, dataclasses, depth) is the one of
   `coola.recursive`. `batchtensor` does not re-implement it.
4. **Follow `CLAUDE.md`.** Batch is dim 0 and sequence is dim 1. `tensor` and `nested` stay in sync.
   Options are keyword-only. Shared helpers are public functions of `batchtensor.utils`. Optional
   libraries are imported lazily.
5. **No slower than today.** A nested call stays dominated by the torch call (Section 6.2).
6. **A claim has a gate.** A property that is written in the docs is enforced by a test or a CI job
   that fails when it breaks. A property without a gate is either given one or removed.

## 1. Properties of the package

Each property has an owner (the place that enforces it), a gate (the test or job that fails when
the property breaks) and a status on `main`.

| #   | Property                  | Statement                                                                                                                                                                                                      | Enforced by                        | Gate on `main`                                          |
| --- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------- |
| P1  | Fixed layout              | Batch is dim 0 and sequence is dim 1. The dims come only from `BATCH_DIM` and `SEQ_DIM`. No function of `tensor` or `nested` takes a `dim` argument. `utils.structure` is dim-generic (it is the layer below). | `constants.py`                     | **Missing** (G2)                                        |
| P2  | Thin wrappers             | A `tensor` function is one torch call with the dim fixed (`masked_*`, Section 4.1, is the one documented exception: a fixed composition). Same semantics as torch. No hidden copy, cast or device move.        | `tensor/*`                         | `test_torch_reference`                                  |
| P3  | Two symmetric levels      | Every `tensor` function that maps batch data has a `nested` twin with the same name, options and defaults. The exceptions are listed in the parity test.                                                       | Parity test                        | `test_api_parity`                                       |
| P4  | One traversal             | All nested functions go through `apply_to_tensors`, `split_tensors` and `combine_tensors`. No nested function walks a container itself.                                                                        | `utils/structure.py`               | **Missing** (G2)                                        |
| P5  | Structure-preserving      | A map function returns the input structure with every leaf replaced. Container types, keys and order follow `coola.recursive` (dict, list, tuple, named tuple, `OrderedDict`, `defaultdict`, dataclass).       | `apply_to_tensors`                 | `test_differential`, `test_structure`                   |
| P6  | Tensor leaves only        | A leaf that is not a `torch.Tensor` raises `TypeError` that names the leaf type. Nothing is skipped silently. The only exception is `from_numpy`, whose leaves are arrays.                                     | `apply_to_tensors`                 | `test_non_tensor_leaf` (**`to_numpy` not covered**, G3) |
| P7  | Pure                      | Inputs are never modified. A function returns a view where torch returns a view (documented per function), otherwise a new tensor.                                                                             | Thin wrappers                      | **Missing** (G2)                                        |
| P8  | One error contract        | A bad leaf type is a `TypeError`. An argument that `batchtensor` validates is a `ValueError`. Anything else is the torch error, unchanged.                                                                     | Validators in `utils`, `tensor`    | Per-function unit tests (no table test)                 |
| P9  | Explicit randomness       | Randomness comes only from an explicit `generator` (a `torch.Generator` or an `int` seed). Without it, the global torch RNG is used. One permutation is shared by all the leaves.                              | `utils.seed.setup_torch_generator` | `test_seed`, `test_device`                              |
| P10 | Device and dtype agnostic | No function moves data between devices or casts dtype (type promotion that torch does is torch's). No host-device sync, except `lengths_to_mask` without `max_len`.                                            | Thin wrappers                      | `test_device` (skipped without an accelerator)          |
| P11 | Differentiable            | Autograd flows through every function whose torch op is differentiable. Only `to_numpy` and `detach` cut the graph (documented).                                                                               | Thin wrappers                      | `test_autograd`                                         |
| P12 | Keyword-only options      | Positional arguments are the data and the arguments that define the operation. Options are keyword-only. No `**kwargs` forwarding, except `nested.to`.                                                         | Signatures                         | `test_signatures`                                       |
| P13 | Typed                     | A map function has the type `T -> T` with `T` bound to `NestedTensor`. Dict in, dict out, tensor in, tensor out. Checked by both `ty` and `pyright`.                                                           | Annotations                        | `ty` in CI; **`pyright` not in CI** (G1); `test_typing` |
| P14 | Layered                   | `constants` <- `utils` <- `tensor` <- `nested`. No import in the other direction. `torch` and `coola` are the only required dependencies. `numpy` is imported lazily.                                          | `import-linter` contract           | `lint-imports` (`ci-cyclic-imports`)                    |
| P15 | Documented by example     | Every public function has a docstring with a runnable `pycon` example. Each property is stated once, in `design_principles.md`.                                                                                | Docstrings                         | `ci-doctest`                                            |

**Not guaranteed:**

- Shared references. Two keys that point to the same tensor are processed twice and give two
  outputs (verified: `clone({"a": x, "b": x})` returns two tensors).
- Memory sharing between input and output, except for the functions documented as views.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`shuffle_*`, `chunk_*`,
  `split_*`).
- Static typing of dataclass inputs: dataclasses work at runtime but are rejected by both type
  checkers (G5).
- Container-type agreement in `combine_tensors`: the output has the container types of the first
  item. A list next to a tuple is accepted.

## 2. What differs today

The surface is 68 names in `tensor` and 74 in `nested` (73 functions and the `NestedTensor` alias,
Appendix A). The code is about 6.8k lines in `src` and 8.3k in `tests`, most of `src` being
docstrings. All the findings of the first version are fixed. The remaining findings are about
claims that have no gate, a few edges, and missing features.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                               | Tag        | Property   |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| G1  | **`pyright` is configured but never run in CI.** Decision D9 says both checkers must pass. `ci-type-checking.yaml` runs `ty check` only. `pyright` is a dev dependency and passes locally (0 errors). The claim "checked by `ty` and `pyright`" in `test_typing.py` is not enforced.                                                                                                                                  | [verified] | P13        |
| G2  | **Three properties have no gate.** P1 (no `dim` parameter anywhere in `tensor` and `nested`), P4 (no nested function touches containers) and P7 (inputs not modified) are in `design_principles.md`, which says "Each principle below is enforced by a test". `test_signatures` checks P12 only. No integration test checks P1, P4 or P7 (P7 is tested only for the helpers in `test_structure`).                     | [verified] | P1, P4, P7 |
| G3  | **`to_numpy` breaks P6 without saying so.** `to_numpy({"a": 3})` raises `AttributeError: 'int' object has no attribute 'detach'`. It uses `recursive_apply` directly. Only `from_numpy` has a reason to accept non-tensor leaves (its leaves are arrays; torch raises its own `TypeError`).                                                                                                                           | [verified] | P6, P8     |
| G4  | **`get_size(data, dim)` takes a `dim`** and is the only way to read the batch size of a nested structure. There is no `nested` function for the question users ask most ("how many samples?"). The raw `dim` invites hard-coded `0`.                                                                                                                                                                                  | [verified] | P1         |
| G5  | **Dataclasses are accepted at runtime and rejected by the type checkers.** `sum_along_batch(D(x))` runs; `ty` and `pyright` both report `D` is not assignable to `NestedTensor`. The docs list dataclasses as supported. `cat`/`stack` reject them at runtime with `TypeError: Expected a torch.Tensor leaf but received D`, which does not say that dataclasses are unsupported there (`chunk`/`split` accept them). | [verified] | P5, P13    |
| G6  | **`utils.structure` is untyped on the output side.** `T = TypeVar("T")` is unbound and `fn` returns `Any`, while `nested.types.T` is bound to `NestedTensor`. Two `TypeVar` named `T`, in two layers, with different meaning. `max`/`min`/`median`/`sort`/`topk` and `chunk`/`split`/`cat` fall back to `NestedTensor`/`Any`.                                                                                         | [reading]  | P13        |
| G7  | **`pad_sequences` has no `nested` twin although `combine_tensors` makes it 3 lines.** Building a batch from a list of samples `[{"x": t1, "mask": m1}, {"x": t2, "mask": m2}]` is the main use of the function and is exactly the `combine` shape. Today the user pads each key by hand. `lengths_to_mask` is the only legitimate parity exception.                                                                   | [reading]  | P3         |
| G8  | **No masked reduction.** The most common sequence operation on a `(batch, seq, ...)` tensor with padding (`mean`/`sum`/`max` over the valid steps) needs `lengths_to_mask` + manual `masked_fill` and a division. D10 deferred it "for its own design". The design is in Section 4.1.                                                                                                                                 | [verified] | P3         |
| G9  | **Boilerplate grows linearly with the operations.** Each operation costs four public functions (2 levels x 2 dims), four docstrings (about 35 lines each) and four test blocks. The eight operations of D10 added 32 functions. This is the product (documented examples), but it sets a limit on how many operations are worth adding.                                                                               | [reading]  | P15        |
| G10 | **Smaller edges.** `split_*` with sizes that do not sum to the dim gives torch's `RuntimeError` (kept by P8). `chunk_*(data, 0)` gives torch's `RuntimeError` (kept). `pad_sequences` with mixed dtypes promotes to the widest (`float32` for float + long; torch's rule, undocumented in the docstring). `shuffle_*` on `{}` raises `ValueError` while `sum_*` on `{}` returns `{}`.                                 | [verified] | P8, P10    |
| G11 | **`coola.recursive.rebuild` is imported by its module path.** `rebuild_mapping` and `rebuild_sequence` are not in `coola.recursive.__all__`. It works, but it is a private-looking import of a sibling package (decision D8, still open).                                                                                                                                                                             | [verified] | P4         |

What is **already right** and stays: the thin `tensor` layer, `apply_to_tensors` as the single
choke point for the leaf rule, the `ValueError` contract for validated arguments, keyword-only
options with a signature gate, the parity and differential tests, the `import-linter` contract,
the named-tuple results of `max`/`min`/`median`/`sort`/`topk`, the migration guide and the
rewritten `design_principles.md`.

## 3. Architecture

The layout does not change.

```text
batchtensor/
  constants.py        BATCH_DIM = 0, SEQ_DIM = 1
  utils/
    seed.py           setup_torch_generator(generator_or_seed)
    structure.py      apply_to_tensors, split_tensors, combine_tensors, get_size
    types.py          NestedTensor, T                                         (moved here, G6)
  tensor/             one torch call per function, dim fixed
    comparison  indexing  joining  masking  math  padding  permutation
    reduction   slicing
  nested/             one helper call per function, same names as tensor/
    comparison  indexing  joining  math  padding  permutation  reduction
    slicing     misc  conversion  size                                      (size: G4)
```

Layering (enforced, P14): `constants` <- `utils` <- `tensor` <- `nested`.

### 3.1 One definition of the nested types (G6)

`NestedTensor` and `T` are defined once, in `batchtensor.utils.types`, because `utils.structure`
needs them and `utils` cannot import `nested`. `batchtensor.nested` re-exports `NestedTensor`
(public) and `batchtensor.nested.types` is removed. The helpers get the same typing as the
functions that call them:

```python
def apply_to_tensors(data: T, fn: Callable[[Tensor], Any]) -> T: ...
def split_tensors(data: T, fn: Callable[[Tensor], tuple[Tensor, ...]]) -> tuple[T, ...]: ...
def combine_tensors(items: Sequence[T], fn: Callable[[list[Tensor]], Tensor]) -> T: ...
```

`apply_to_tensors` keeps `fn: Callable[[Tensor], Any]` and its `T -> T` signature: the helper cannot
check the output type of `fn`, so the nested function that calls it is responsible for the
annotation. The five families whose leaf becomes a named tuple (`max`, `min`, `median`, `sort`,
`topk`) annotate `NestedTensor -> NestedTensor` and the others `T -> T`. The target of this step is
only to have one `T` and one `NestedTensor`, and to type `split_tensors` and `combine_tensors` with
`T` instead of `Any` (D11 records the alternative).

### 3.2 A nested function stays logic-free

```python
def sum_along_batch(data: T, *, keepdim: bool = False) -> T:
    return apply_to_tensors(data, partial(bt.sum_along_batch, keepdim=keepdim))
```

A nested function that needs logic puts it in a helper, tested once. The nested functions that
call `get_size` for validation (`shuffle_*`) are the only ones with two lines, and they go through
the public `batch_size`/`seq_len` (Section 4.2) instead of `get_size(data, DIM)`.

### 3.3 Conversion functions

`to_numpy` uses `apply_to_tensors` (G3): its leaves must be tensors, like every other function.
`from_numpy` keeps `recursive_apply(data, torch.from_numpy)`: its leaves are arrays, the leaf type
error is torch's `TypeError`, and this is the one documented exception of P6. `numpy` stays a lazy
import in both.

### 3.4 Dataclasses (G5)

Decision D4 stays: `split_tensors` and `combine_tensors` do not rebuild dataclasses differently,
and `coola.recursive.rebuild` has no `rebuild_dataclass`. The target is to make the situation
explicit, not to widen it:

- the docs say: dataclasses are supported by every map function and by `chunk`/`split` at runtime,
  not by `cat`/`stack`, and the type aliases do not include them;
- `combine_tensors` raises `TypeError("... dataclasses are not supported by combine_tensors; got D")`
  for a dataclass (one `dataclasses.is_dataclass` check, one test);
- `NestedTensor` stays strict, and the docs say dataclasses are untyped: a dataclass in/out cannot
  be expressed by `T -> T` with a bound, and widening the bound to a protocol would weaken the
  checks for every other input.

## 4. Public API after the change

### 4.1 `tensor`

| Function(s)                                                                                                             | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| All current functions (68 names)                                                                                        | No change.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `pad_sequences`                                                                                                         | Documents the dtype rule: the result dtype follows torch's type promotion (G10). No signature change.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **`masked_sum_along_seq`**, **`masked_mean_along_seq`**, **`masked_amax_along_seq`**, **`masked_amin_along_seq`** (new) | `f(tensor, mask, *, keepdim=False)`. `mask` is a `bool` tensor of shape `(batch, seq)` (what `lengths_to_mask` returns), `True` for a valid step. It is broadcast over the trailing dims of `tensor`. Fixed compositions: `sum`: `tensor.masked_fill(~m, 0).sum(1)`. `mean`: the `sum` divided by the number of valid steps. `amax`/`amin`: `masked_fill` with the dtype's lowest/highest value, then `amax`/`amin`. An all-`False` row gives `0` for `sum`, `nan` for `mean` (0/0, like `torch.mean` of an empty tensor) and the sentinel for `amax`/`amin`. Raises `ValueError` if the mask is not `bool`, not 2-D, or does not match the first two dims. |
| `lengths_to_mask`                                                                                                       | No change. It stays the producer of the masks above.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

The masked family is the only breach of P2 and is deliberate: four functions, `_along_seq` only
(a masked batch reduction has no use case), documented as compositions, with a differential test
against the manual `masked_fill` form. `mask` is a positional argument because it defines the
operation (P12). A `masked_*_along_batch` is not added (Section 10).

`tensor.__all__` goes from 68 to 72 names.

### 4.2 `nested`

| Function(s)                               | Change                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Every twin of a `tensor` function         | No change.                                                                                                                                                                                                                                                                                                                                                   |
| **`masked_*_along_seq`** (new)            | Same name and options as in 4.1. `mask` is **one tensor shared by all the leaves** (not a structure): `apply_to_tensors(data, partial(bt.masked_mean_along_seq, mask=mask))`. Every leaf must have at least two dims and a `(batch, seq)` prefix equal to the mask shape.                                                                                    |
| **`pad_sequences`** (new, G7)             | `pad_sequences(samples: Sequence[T], *, value=0.0, side="right") -> T`. Pads each leaf position across the samples with `combine_tensors(samples, partial(bt.pad_sequences, value=value, side=side))`. Samples are structures with the same layout; the leaves have different lengths along dim 0. The parity exception list becomes `lengths_to_mask` only. |
| **`batch_size`**, **`seq_len`** (new, G4) | `batch_size(data) -> int` is `get_size(data, BATCH_DIM)` and `seq_len(data) -> int` is `get_size(data, SEQ_DIM)`. They accept any structure that `apply_to_tensors` accepts and raise like `get_size`. Nested-only by design (they return an `int`, not a structure); listed as nested-only in the parity test.                                              |
| `to_numpy`                                | Goes through `apply_to_tensors` (G3). A non-tensor leaf raises `TypeError`. A behaviour change (`AttributeError` to `TypeError`).                                                                                                                                                                                                                            |
| `from_numpy`                              | No change. The documented exception of P6.                                                                                                                                                                                                                                                                                                                   |
| `types.NestedTensor`                      | Moved to `batchtensor.utils.types`, still re-exported from `batchtensor.nested` (G6). `batchtensor.nested.types` is removed.                                                                                                                                                                                                                                 |

`nested.__all__` goes from 74 to 81 names (`masked_*` x4, `pad_sequences`, `batch_size`,
`seq_len`; `NestedTensor` is counted in both numbers).

### 4.3 `utils` and `constants`

| Name                                               | Change                                                                             |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `utils.structure.apply_to_tensors`                 | Keeps `T -> T` with an `Any`-returning `fn`; uses the single `T` (G6, D11).        |
| `utils.structure.split_tensors`, `combine_tensors` | Typed with `T`. `combine_tensors` raises a clear `TypeError` for a dataclass (G5). |
| `utils.structure.get_size`                         | Unchanged. Documented as dim-generic (P1 applies to `tensor` and `nested`).        |
| `utils.types.NestedTensor`, `T`                    | **New location** (moved from `nested.types`).                                      |
| `utils.seed.setup_torch_generator`, `constants.*`  | Unchanged.                                                                         |

## 5. Dependents

- **`coola`.** `batchtensor` depends on `coola.recursive.recursive_apply`, `coola.iterator.dfs_iterate`
  and `coola.recursive.rebuild.rebuild_mapping`/`rebuild_sequence`. Ask for the `rebuild_*` functions
  to be exported from `coola.recursive` (D8, G11) before the first release.
- **Users.** The breaking changes of this plan are small: `to_numpy` error type, the import path
  `batchtensor.nested.types` (the public path `batchtensor.nested.NestedTensor` is unchanged), and
  the `masked_*`, `pad_sequences`, `batch_size`, `seq_len` additions. They are added to
  `docs/docs/migration.md`.

## 6. Evidence

### 6.1 Probes (run against `main`, `1009e2f`)

| Probe                                                                  | Result on `main`                                                                                                                                                |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.venv/bin/pytest tests`                                               | 1 812 passed, 3 skipped (device tests without an accelerator)                                                                                                   |
| `ruff check`, `ty check`, `lint-imports`, `pyright src`                | all clean                                                                                                                                                       |
| `grep pyright .github Makefile tasks.py`                               | no match: not run in CI (G1)                                                                                                                                    |
| `tensor.__all__` / `nested.__all__` difference                         | `tensor` only: `lengths_to_mask`, `pad_sequences`; `nested` only: `NestedTensor`, `clone`, `contiguous`, `detach`, `from_numpy`, `pin_memory`, `to`, `to_numpy` |
| `to_numpy({"a": 3})`                                                   | `AttributeError: 'int' object has no attribute 'detach'` (G3)                                                                                                   |
| `from_numpy({"a": [1, 2]})`                                            | `TypeError: expected np.ndarray (got int)`                                                                                                                      |
| `cat_along_batch([D(x), D(x)])` with a dataclass `D`                   | `TypeError: Expected a torch.Tensor leaf but received D` (G5)                                                                                                   |
| `chunk_along_batch(D(x), 2)`                                           | two `D` (dataclass entered)                                                                                                                                     |
| `sum_along_batch(D(x))` with `ty` and `pyright`                        | both report `D` not assignable to `NestedTensor` (G5)                                                                                                           |
| `cat_along_batch([[x], (x,)])`                                         | accepted, returns the type of the first item                                                                                                                    |
| `cat_along_batch([NT(x, x)] * 2)`, `chunk_along_batch(NT(x, x), 2)[0]` | `NT` kept                                                                                                                                                       |
| `cat_along_batch([{"a": x, "b": x}, {"b": x, "a": x}])` key order      | `['a', 'b']` (order of the first item)                                                                                                                          |
| `stack_along_batch([x, x[:2]])`                                        | torch's `RuntimeError` (P8)                                                                                                                                     |
| `split_along_batch({"a": x}, [3, 3])` on 4 rows                        | torch's `RuntimeError` (P8)                                                                                                                                     |
| `chunk_along_batch({"a": x}, 0)`                                       | torch's `RuntimeError` (P8)                                                                                                                                     |
| `shuffle_along_batch({"a": x, "b": x[:2]})`                            | `ValueError: All the tensors must have the same size along dimension 0`                                                                                         |
| `shuffle_along_seq({"a": arange(3)})`                                  | `ValueError: Expected a tensor with at least 2 dimension(s)`                                                                                                    |
| `shuffle_along_batch({})`, `sum_along_batch({})`                       | `ValueError`, `{}` (G10)                                                                                                                                        |
| `sum_along_batch({"a": "s"})`                                          | `TypeError: Expected a torch.Tensor leaf but received str`                                                                                                      |
| `shuffle_along_batch(x, generator=0)` twice                            | same result                                                                                                                                                     |
| `pad_sequences([float32 (2,), long (3,)]).dtype`                       | `float32` (G10)                                                                                                                                                 |
| `clone({"a": x, "b": x})`                                              | two distinct tensors (shared reference not kept)                                                                                                                |
| `max_along_batch({"a": x})["a"]`, `topk_along_batch(...)["a"]`         | `torch.return_types.max`, `torch.return_types.topk`                                                                                                             |
| nested `NestedTensor` type for `[[x]]` of `str`                        | `str`/`bytes` excluded                                                                                                                                          |

### 6.2 Cost

macOS, Python 3.14, CPU, a dict of 10 tensors of shape `(8, 4)`, mean of 2 000 calls (a single run;
the numbers give an order of magnitude, not a benchmark):

| Call                                    | Time    |
| --------------------------------------- | ------- |
| manual `{k: bt.sum_along_batch(v) ...}` | 11.9 us |
| `bn.sum_along_batch(data)`              | 16.3 us |
| `bn.cat_along_batch([data, data])`      | 14.7 us |
| `bn.chunk_along_batch(data, 2)`         | 28.1 us |
| `bn.shuffle_along_batch(data)`          | 30.3 us |

The helper layer costs 1.4x a hand-written loop on 10 leaves and is below the 26.9 us of the
version before the refactoring. No performance work is needed. The gate is: the same script must
not regress by more than 20 % against `main`.

### 6.3 Typing probe

`ty` and `pyright` accept `nested.sum_along_batch(tensor) -> Tensor` and
`dict[str, Tensor] -> dict[str, Tensor]` (`test_typing.py`). Both reject a dataclass argument (G5).

## 7. Changes to reach the target state

One branch and one PR per step, stacked when a step depends on the previous one (`CLAUDE.md`).
Steps 1 to 11 of the first version are merged (Appendix B). The new steps continue the numbering.

| Step | Branch                      | Content                                                                                                                                                                                                                                                                                                                                                               | Breaking                  | Finding |
| ---- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------- |
| 12   | `ci/run-pyright`            | Add `pyright` to `ci-type-checking.yaml` (or record the decision to drop it, D9). Fix what it reports (nothing locally).                                                                                                                                                                                                                                              | No                        | G1      |
| 13   | `test/property-gates`       | Add the three missing gates: no `dim` parameter in `tensor`/`nested` (signature test), no container walk in `nested` (AST test: no `isinstance(..., (dict, list, tuple, Mapping))` and no `recursive_apply` import), no input mutation (one table over `tensor.__all__` and `nested.__all__`: clone inputs, call, compare). Add `to_numpy` to `test_non_tensor_leaf`. | No                        | G2      |
| 14   | `bugfix/to-numpy-leaf`      | `to_numpy` goes through `apply_to_tensors`. Regression test with `match`, checked to fail without the fix.                                                                                                                                                                                                                                                            | Yes (error type)          | G3      |
| 15   | `refactor/nested-types`     | Move `NestedTensor`/`T` to `utils/types.py`, type the three helpers with `T`, re-export `NestedTensor`, remove `nested/types.py`. Make `combine_tensors` raise a clear error for a dataclass. Document the dataclass situation in `design_principles.md` and `uguide/nested.md`.                                                                                      | Yes (`nested.types` path) | G5, G6  |
| 16   | `feat/batch-size-seq-len`   | Add `nested.batch_size`, `nested.seq_len`. Use them in `shuffle_*`. Add them to the nested-only list of the parity test.                                                                                                                                                                                                                                              | No                        | G4      |
| 17   | `feat/nested-pad-sequences` | Add `nested.pad_sequences` through `combine_tensors`. Parity exception list becomes `lengths_to_mask`. Document the dtype rule of `pad_sequences`.                                                                                                                                                                                                                    | No                        | G7, G10 |
| 18   | `feat/masked-reductions`    | Add `masked_{sum,mean,amax,amin}_along_seq` to `tensor` and `nested`, with validation, differential test against the manual form, autograd test and a user-guide section.                                                                                                                                                                                             | No                        | G8      |
| 19   | `chore/coola-rebuild`       | When `coola` exports `rebuild_mapping`/`rebuild_sequence` from `coola.recursive`, change the import. Until then, keep the module path.                                                                                                                                                                                                                                | No                        | G11     |

Steps 12 to 14 are independent of each other and small. Step 15 comes before 16 to 18 (the new
nested functions use the typed helpers). Steps 16 to 18 are independent after 15. Step 19 depends
on `coola` only.

## 8. Tests and gates

**Unit tests** (mirror the tree):

- `utils/test_structure.py`: `combine_tensors` with a dataclass (`TypeError` with `match`), with
  a list next to a tuple (first item's type), with dicts in different key order.
- `utils/test_types.py`: none (no trivial structure test). The alias is checked by `test_typing`.
- `tensor/test_masking.py`: `masked_sum/mean/amax/amin_along_seq` for a typical mask, an all-`False`
  row, an all-`True` mask, a 2-D tensor, a 4-D tensor (trailing dims), float and int dtypes (`mean`
  on int is torch's error), `keepdim`, a non-`bool` mask, a 1-D mask, a mask with the wrong shape,
  no input mutation, the whole output against a literal.
- `nested/test_masking.py`: one parametrized case per function (dict, list, tuple), the shared mask,
  a leaf that does not match the mask.
- `nested/test_padding.py`: `pad_sequences` with dicts, lists, ragged leaves, `side="left"`,
  different keys (`ValueError`), an empty sequence (`ValueError`), a non-tensor leaf.
- `nested/test_size.py`: `batch_size`, `seq_len` for a tensor, a dict, a list, a mixed nest; no
  tensor, disagreeing leaves, a 1-D leaf for `seq_len`.
- `nested/test_conversion.py`: `to_numpy` with a non-tensor leaf.

**Integration tests** (`tests/integration/`, no patching):

- `test_properties.py` (step 13): P1 (no parameter named `dim` in any public signature of `tensor`
  and `nested`), P4 (the AST check), P7 (no mutation, one parametrized table).
- `test_api_parity.py`: extended allow-lists (`lengths_to_mask` tensor-only; `batch_size`,
  `seq_len` and the lifecycle/conversion functions nested-only).
- `test_differential.py` and `test_torch_reference.py`: add the masked family (compared with
  the manual `masked_fill` form, not a torch call, since P2 has no single torch call to match).
- `test_pipeline.py`: collate with `nested.pad_sequences` on a list of dicts, then `lengths_to_mask`,
  `masked_mean_along_seq`, `shuffle`, `chunk` on the same batch.
- `test_typing.py`: add `masked_mean_along_seq`, `pad_sequences`, `batch_size` (`assert_type`).

**Gates (CI):**

- `ruff check`, `ruff format`, `ty check`, **`pyright`** (G1), `lint-imports`.
- Unit and integration tests on the oldest supported `torch` (`2.4`, job `min`) and the latest.
- 100 % coverage of `src` without `pragma: no cover` that could be tested.
- Doctest job on every docstring.
- Benchmark script of Section 6.2: no regression greater than 20 %.
- Regression test for every bug fixed (G3), verified to fail without the fix
  (`git stash push src`).

## 9. Decisions for the maintainer

Decisions D1 to D10 of the first version are closed and applied (Appendix B). New decisions:

| #   | Question                                                                                                 | Recommendation                                                                                                                                                                                                                                                                                                                                    |
| --- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D9  | (reopened) Keep both `ty` and `pyright`?                                                                 | The previous decision is **keep both**, and CI does not honour it (G1). Recommended: honour it with one more CI step (step 12). Alternative: drop `pyright` and the "checked by both" claims; one checker is simpler and the project already runs `ty`. Do one of the two.                                                                        |
| D11 | Type the output of `apply_to_tensors` more precisely (a `Tensor`-returning `fn`)?                        | **No.** Keep `T -> T` with `fn` returning `Any`; the five named-tuple families annotate `NestedTensor` by hand. A second helper only to carry a type adds a name to learn for no runtime gain. Alternative: `apply_to_tensors` (`fn -> Tensor`, `T -> T`) plus `map_to_tensors` (`fn -> R`, `NestedTensor -> NestedTensor`), exact but two names. |
| D12 | Masked reductions: `mask` argument, a `masked_` prefix, or a keyword `mask=` on the existing reductions? | **Separate `masked_*_along_seq` functions.** A `mask=None` option on `mean_along_seq` breaks P2 (one torch call) for the whole family and doubles the validation surface. Alternative: no masked reductions in the package (users keep the manual form).                                                                                          |
| D13 | Is the mask of a nested masked reduction a structure or one tensor?                                      | **One tensor shared by all the leaves.** It matches `lengths_to_mask` and needs no zip of two structures (`coola` has no multi-input apply). A per-leaf mask is a loop over `apply_to_tensors` in user code.                                                                                                                                      |
| D14 | Add `batch_size`/`seq_len` to `nested`, or keep only `utils.structure.get_size`?                         | **Add.** They are read-only helpers, 3 lines each, they remove the raw `0`/`1` from user code and keep P1 true for `nested`. Alternative: document `get_size(data, dim=BATCH_DIM)` as the answer.                                                                                                                                                 |
| D4  | (unchanged) Dataclasses in `combine_tensors` and `split_tensors`                                         | **Not now**, but make the failure explicit (step 15). Revisit when `coola` has a dataclass rebuild or a multi-input apply.                                                                                                                                                                                                                        |
| D8  | (unchanged) Use `coola.recursive.rebuild` from `batchtensor`                                             | **Yes**, and ask `coola` to export the `rebuild_*` functions (step 19).                                                                                                                                                                                                                                                                           |

## 10. Not covered

- Masked reductions along the batch dim (`masked_*_along_batch`): no use case identified.
- Further masked operations (`softmax`, `argmax`, `topk`) and a `where`/`masked_fill` family. Add
  them on request, one PR each, with the rule of G9.
- A table-driven or generated definition of the 4-function families (G9). The documented example
  per function is the product; generated docstrings lose it.
- Sequence-first or channel-first layouts, a configurable `seq_dim`.
- Dataclass support in `combine_tensors`/`split_tensors` and in the static types (D4, G5).
- Reporting the path of a bad leaf in the `TypeError` (needs a path-aware `coola` engine).
- Sparse, jagged or masked `torch` tensors, and `numpy`/`pandas` leaves.
- A `torch.compile` audit of the wrappers.
- Release engineering (version table, classifiers, nightly jobs).

## Appendix A. API inventory

| Family                                             | `tensor` now | `tensor` target | `nested` now | `nested` target |
| -------------------------------------------------- | ------------ | --------------- | ------------ | --------------- |
| reduction (15 operations x 2 dims)                 | 30           | 30              | 30           | 30              |
| comparison (`sort`, `argsort`, `topk`)             | 6            | 6               | 6            | 6               |
| math (`cumsum`, `cumprod`)                         | 4            | 4               | 4            | 4               |
| slicing                                            | 8            | 8               | 8            | 8               |
| joining (`cat`, `stack`)                           | 4            | 4               | 4            | 4               |
| indexing, `repeat` (2 operations x 2 dims)         | 4            | 4               | 4            | 4               |
| permutation (`permute`, `shuffle`, `flip`, `roll`) | 8            | 8               | 8            | 8               |
| padding                                            | 2            | 2               | 2            | 2               |
| `pad_sequences`                                    | 1            | 1               | 0            | 1               |
| `lengths_to_mask`                                  | 1            | 1               | 0            | 0               |
| masked reductions (new)                            | 0            | 4               | 0            | 4               |
| `batch_size`, `seq_len` (new)                      | 0            | 0               | 0            | 2               |
| lifecycle (`clone`, `detach`, ...)                 | 0            | 0               | 5            | 5               |
| conversion (`to_numpy`, `from_numpy`)              | 0            | 0               | 2            | 2               |
| `NestedTensor`                                     | 0            | 0               | 1            | 1               |
| **Total**                                          | **68**       | **72**          | **74**       | **81**          |

(The `now` columns are `len(__all__)` on `main`, counted by operation name. The reductions are
`sum`, `mean`, `prod`, `amax`, `amin`, `argmax`, `argmin`, `max`, `min`, `median`, `std`, `var`,
`any`, `all`, `logsumexp`.)

## Appendix B. Record of the first target state

| Step | Content                                                              | PR                                             |
| ---- | -------------------------------------------------------------------- | ---------------------------------------------- |
| 1    | Fix the documentation contradictions                                 | #737                                           |
| 2    | Enforce the layering with `import-linter`                            | #738                                           |
| 3    | `utils/structure.py` helpers, `T -> T` typing, public `NestedTensor` | #739                                           |
| 4    | Keyword-only options                                                 | #722                                           |
| 5    | Remove the pointwise functions and `as_tensor`                       | #723                                           |
| 6    | `int` seed in `shuffle_*`                                            | #724                                           |
| 7    | Validation contract (`ValueError`)                                   | #725                                           |
| 8    | `pad` side, `repeat_along_batch`, `pad_sequences`                    | #726                                           |
| 9    | `design_principles.md` and the migration guide                       | #728                                           |
| 10   | API parity, differential, autograd, device tests                     | #727                                           |
| 11   | `std`, `var`, `any`, `all`, `logsumexp`, `flip`, `roll`, `topk`      | #729, #730, #731, #740, #741, #742, #743, #744 |

Closed decisions: D1 (remove pointwise: yes), D2 (keep `clone`/`detach`/`contiguous`/`pin_memory`/`to`),
D3 (remove `as_tensor`), D5 (keep the 2-D `index_select_along_seq`), D6 (keep `permute_along_*`),
D7 (named-tuple leaves), D10 (add the eight operations).
