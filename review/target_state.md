# `batchtensor`: target state and the changes to reach it

**Scope.** This document proposes a target state for the whole `batchtensor` package on `main`
(`constants`, `utils`, `tensor`, `nested`) and the changes to reach it. It changes no code.
It is based on a read of the sources, the documentation and the tests, and on probes run against
`main` with `coola 2.0.0a2` and `torch` from `.venv` (Python 3.14). Section 7 lists the probes.
It supersedes `review/suggestions.md`, which is partly stale (Appendix A gives the status of each
item).

This is a pre-1.0 release (`0.1.2a1`) and the README already says that the API is not stable.
The target is a breaking release.

## Constraints

1. **Simple to maintain.** Every function is a thin wrapper. A change is made in one place.
   Remove code instead of adding workarounds.
2. **Breaking changes are allowed.** No deprecation period and no adapters. The future state matters
   more than the past state.
3. **Consistent with `coola`.** Container handling (types, keys, dataclasses, depth) is the one of
   `coola.recursive`. `batchtensor` does not re-implement it.
4. **Follow `CLAUDE.md`.** Batch is dim 0 and sequence is dim 1. `tensor` and `nested` stay in sync.
   Options are keyword-only. Shared helpers are public functions of `batchtensor.utils`.
   Optional libraries are imported lazily.
5. **No slower than today.** Nested calls stay dominated by the torch call (Section 7.2).

## 1. Properties of the package

Each property has an owner (the place that enforces it) and a gate (the test that fails when the
property is broken).

| #   | Property                  | Summary                                                                                                                                                                           | Enforced by                        | Gate                        |
| --- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | --------------------------- |
| P1  | Fixed layout              | Batch is dim 0 and sequence is dim 1. The dims come only from `BATCH_DIM` and `SEQ_DIM`. No function takes a `dim` argument.                                                      | `constants.py`                     | Layout tests                |
| P2  | Thin wrappers             | A `tensor` function is one torch call with the dim fixed. Same semantics and same result as torch. No hidden copies, casts or device moves.                                       | `tensor/*`                         | Differential tests vs torch |
| P3  | Two symmetric levels      | Every `tensor` function that maps batch data has a `nested` twin with the same name, the same options and the same defaults. Exceptions are listed (Section 4).                   | Parity test                        | `test_api_parity`           |
| P4  | One traversal             | All nested functions go through three helpers in `batchtensor.utils`. No function walks a container itself.                                                                       | `utils/structure.py`               | Import and AST test         |
| P5  | Structure-preserving      | A map function returns the input structure with every leaf replaced. Container types, keys and order are kept (the `coola.recursive` rules).                                      | `apply_to_tensors`                 | Structure tests             |
| P6  | Tensor leaves only        | A leaf that is not a `torch.Tensor` raises `TypeError` that names the leaf type. Nothing is skipped silently. Conversion functions are the documented exception.                  | `apply_to_tensors`                 | Leaf tests                  |
| P7  | Pure                      | Inputs are never modified. A function returns a view where torch returns a view (documented per function), otherwise a new tensor.                                                | Thin wrappers                      | No-mutation tests           |
| P8  | One error contract        | A bad leaf type is a `TypeError`. An argument that `batchtensor` validates is a `ValueError`. Anything else is the torch error, unchanged.                                        | Validators in `utils`, `tensor`    | Error table tests           |
| P9  | Explicit randomness       | Randomness comes only from an explicit `generator` (a `torch.Generator` or an `int` seed). Without it, the global torch RNG is used. One permutation is shared by all the leaves. | `utils.setup_torch_generator`      | Seed tests                  |
| P10 | Device and dtype agnostic | No function moves data between devices or casts dtype. No host-device sync, except where documented (`lengths_to_mask` without `max_len`).                                        | Thin wrappers                      | Docstring + device test     |
| P11 | Differentiable            | Autograd flows through every function whose torch op is differentiable. Only `to_numpy` detaches (documented).                                                                    | Thin wrappers                      | One autograd test           |
| P12 | Keyword-only options      | Positional arguments are the data and the arguments that define the operation. Options (`keepdim`, `descending`, `value`, `generator`, ...) are keyword-only.                     | Signatures                         | Signature test              |
| P13 | Typed                     | A map function has the type `T -> T` with `T` bound to `NestedTensor`. Dict in, dict out. Tensor in, tensor out.                                                                  | Annotations                        | `ty check` + typing tests   |
| P14 | Layered                   | `constants` <- `utils` <- `tensor` <- `nested`. No import in the other direction. `torch` and `coola` are the only required dependencies. `numpy` is never imported.              | `import-linter` contract           | `lint-imports`              |
| P15 | Documented by example     | Every public function has a docstring with a runnable `pycon` example. The docs state the contract of each property once.                                                         | Docstrings, `design_principles.md` | Doctest CI job              |

**Not guaranteed:**

- Shared references. Two keys that point to the same tensor are processed twice and give two
  outputs.
- Memory sharing between input and output, except for the functions documented as views
  (`slice`, `select`, `chunk`, `split`, `from_numpy`, `to_numpy` on CPU).
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`shuffle_*`, `chunk_*`,
  `split_*`, Section 3.3).

## 2. What differs today

The number of public functions is 50 in `tensor` and 80 in `nested`. The sources are about 5.9k lines,
most of them docstrings. The code is in good shape: the thin-wrapper idea works and the parity of
signatures between `tensor` and `nested` is exact (probed). The findings are about edges, not about
the core.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                 | Evidence                                                                         | Property |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------- |
| F1  | **Non-tensor leaves fail with a different error per function.** `sum` gives `TypeError` from torch, `slice` gives `TypeError: 'NoneType' is not subscriptable`, `chunk` gives `AttributeError`, `shuffle` gives `AttributeError`. `design_principles.md` says they are "returned unchanged".                                                                                                                                            | Probe `non-tensor leaf *`                                                        | P6, P8   |
| F2  | **Two container engines with different rules.** Map functions use `coola.recursive_apply` (dataclasses entered, `OrderedDict` and `defaultdict` kept, named tuples kept). `combine` and `split` have their own traversal: they return a plain `dict`, a plain `tuple`, lose named tuples, and reject dataclasses.                                                                                                                       | Probes `cat OrderedDict type`, `split ordereddict`, `split preserves namedtuple` | P4, P5   |
| F3  | **`combine([])` returns `{}`.** An empty input is an error, not an empty dict. `split_*({}, n)` returns `()` and hides an empty batch.                                                                                                                                                                                                                                                                                                  | Probes `cat empty`, `split empty dict`                                           | P8       |
| F4  | **`split` and `chunk` do not check that the leaves agree.** `{"a": 3 rows, "b": 2 rows}` is chunked without error and the chunks are misaligned. `shuffle` does check.                                                                                                                                                                                                                                                                  | Probe `split diff batch sizes`                                                   | P8       |
| F5  | **`combine` and `split` are public in `nested` without a `tensor` twin.** They have generic names (`split`, `combine`) next to `split_along_batch`. They are not "along" functions.                                                                                                                                                                                                                                                     | `nested.__all__`                                                                 | P3       |
| F6  | **`as_tensor` is ill-defined on Python lists.** `as_tensor({"a": [1, 2, 3]})` gives a list of three 0-d tensors, not a tensor. The function is also redundant with `from_numpy` for arrays.                                                                                                                                                                                                                                             | Probes `as_tensor list`, `as_tensor nested lists`                                | P5       |
| F7  | **21 element-wise functions (`pointwise.py`, `trigo.py`) have no batch or sequence meaning and no `tensor` twin.** They are `recursive_apply(data, torch.xxx)` with a 25-line docstring. ~590 lines. The module shadows the builtin `abs` (needs `noqa: A001`) and `clamp(min, max)` shadows two more.                                                                                                                                  | `nested/pointwise.py`, `nested/trigo.py`                                         | P3       |
| F8  | **Options are positional-or-keyword.** `keepdim`, `descending`, `value` can be passed positionally. `argsort` and `sort` forward an untyped `**kwargs` to torch (`stable` works by accident, undocumented). `contiguous` and `to` forward `*args, **kwargs`.                                                                                                                                                                            | Signatures; probe `argsort kw`                                                   | P12      |
| F9  | **Validation is inconsistent.** `index_select_along_seq` raises `RuntimeError` for a batch mismatch (the docs say `ValueError`). A 3-D index gives `IndexError: shape mismatch`. A 2-D permutation passes the length check and then fails inside torch. `lengths_to_mask` accepts floats and negative values silently. `shuffle_*(x, 0)` (an int seed) raises `AttributeError`, although `utils.setup_torch_generator` exists for this. | Probes `idx seq *`, `permute 2d perm`, `mask *`, `shuffle seed int`              | P8, P9   |
| F10 | **`permute_along_*` overlaps with `index_select_along_*`.** Both call `index_select` with a 1-D index. `permute` only adds a length check and does not check that the index is a permutation (duplicates pass). Kept by decision D6: the length check and the name document the intent. The name collides with `torch.permute` (which permutes dims), so the docstring says so.                                                                                                                                                          | Probe `permute dup`                                                              | P2       |
| F11 | **`pad_along_*` only pads on the right, and there is no way to build a batch from variable-length tensors.** An integer tensor padded with a float `value` truncates silently (`1.5` becomes `1`). There is no `repeat_along_batch` next to `repeat_along_seq`.                                                                                                                                                                         | Probe `pad int value`                                                            | P3, P8   |
| F12 | **`NestedTensor` hides the result type.** Every nested function returns `NestedTensor`, so a type checker cannot see that a tensor goes in and a tensor comes out. The alias also accepts `str` and `bytes` (they are `Sequence`).                                                                                                                                                                                                      | `nested/types.py`                                                                | P13      |
| F13 | **`utils.setup_torch_generator` is not used in `src`.** The only helper in `utils` is dead in the package. The private `nested.permutation._get_size` is the shared helper of F4, and `CLAUDE.md` asks for shared helpers to be public in `utils`.                                                                                                                                                                                      | `grep`                                                                           | P4, P9   |
| F14 | **The layering is not enforced.** `[tool.importlinter]` has a root package and no contract.                                                                                                                                                                                                                                                                                                                                             | `pyproject.toml`                                                                 | P14      |
| F15 | **The documentation contradicts the code.** See Appendix C (README version table, `pyright`, Pattern 3, "returned unchanged", "flat dict", a stale `suggestions.md`).                                                                                                                                                                                                                                                                   | Appendix C                                                                       | P15      |
| F16 | **The tests have no parity or differential gate.** Parity of `tensor` and `nested` is only checked by hand, one function at a time. `test_types.py` asserts nothing about the alias (`value == value`). `package_checks.py` covers one function per package.                                                                                                                                                                            | `tests/`                                                                         | P3       |
| F17 | **Smaller issues.** `lengths_to_mask` has no `nested` twin and no documented reason. `import-linter`, `pyright` and `ty` are all configured. Both are kept by decision D9, so they must both pass in CI.                                                                                                                                                                                                                                                        | `pyproject.toml`                                                                 | P3, P14  |

What is **already right** and stays: the `tensor` layer (one-liners over torch with the dim constant),
the strict `zip` use, the `dim`-constant use everywhere, the device-safe `shuffle` (the permutation
is created on the generator device and moved to the tensor device), `to_numpy` with
`detach().cpu()`, `repeat_along_seq` validation, the recursive `combine`/`split`, the named-tuple
results of `max`/`min`/`median`/`sort`, and the view-versus-copy documentation.

## 3. Architecture

```text
batchtensor/
  constants.py        BATCH_DIM = 0, SEQ_DIM = 1
  utils/
    seed.py           setup_torch_generator(generator_or_seed)
    structure.py      apply_to_tensors, split_tensors, combine_tensors, get_size   (new)
  tensor/             one torch call per function, dim fixed
    comparison.py  indexing.py  joining.py  masking.py  math.py
    padding.py     reduction.py slicing.py
  nested/             one helper call per function, same names as tensor/
    comparison.py  indexing.py  joining.py  math.py  padding.py
    reduction.py   slicing.py   misc.py     conversion.py  types.py
```

Layering (enforced, P14): `constants` <- `utils` <- `tensor` <- `nested`.
`tensor` imports `constants` and `torch`. `nested` imports `tensor`, `utils` and `coola`.
`utils.structure` imports `coola` and `torch` and nothing else from the package.

### 3.1 The three helpers

All three live in `batchtensor/utils/structure.py`, are public, and have their own tests.

```python
def apply_to_tensors(data: T, fn: Callable[[Tensor], Tensor]) -> T:
    """Apply ``fn`` to every tensor leaf. Raise TypeError for any other leaf."""
    return recursive_apply(data, _guard(fn))  # coola engine: types, keys, depth


def split_tensors(
    data: NestedTensor, fn: Callable[[Tensor], tuple[Tensor, ...]]
) -> tuple[NestedTensor, ...]:
    """Apply ``fn`` to every leaf and return N structures, one per part."""


def combine_tensors(
    items: Sequence[NestedTensor], fn: Callable[[list[Tensor]], Tensor]
) -> NestedTensor:
    """Combine N structures with the same layout into one, leaf by leaf."""
```

- `_guard(fn)` checks `isinstance(leaf, torch.Tensor)` and raises
  `TypeError(f"expected a torch.Tensor leaf but received {type(leaf).__qualname__}")`. This is the
  single choke point of F1. Measured cost is within the noise of the torch call (Section 7.2).
- `split_tensors` runs `fn` once per leaf. The leaf result is wrapped in a small private
  final class (not a tuple and not a dataclass, so the `coola` engine treats it as a leaf).
  The structure is rebuilt N times by reading part `i` of each wrapper. It rejects `n == 0` and
  checks that all leaves return the same number of parts (F4).
- `combine_tensors` keeps the current parallel walk (it is the one place where `coola` has no
  engine: it traverses N structures at once). It changes in three ways: it raises `ValueError` on an
  empty `items` (F3), it rebuilds containers with `coola.recursive.rebuild.rebuild_mapping` and
  `rebuild_sequence` so that `OrderedDict`, `defaultdict` and named tuples keep their type (F2), and
  it raises `TypeError` for a non-tensor leaf (P6). Dataclasses are not supported by `combine`
  and `split` (decision D4).
- `get_size(data, dim)` returns the common size along `dim` of all the leaves. It raises
  `ValueError` if the leaves disagree and `ValueError` if there is no leaf. `shuffle_*`, `chunk_*`
  and `split_*` use it.

### 3.2 A nested function after the change

```python
def sum_along_batch(data: T, *, keepdim: bool = False) -> T:
    return apply_to_tensors(data, partial(bt.sum_along_batch, keepdim=keepdim))


def chunk_along_batch(data: NestedTensor, chunks: int) -> tuple[NestedTensor, ...]:
    return split_tensors(data, partial(bt.chunk_along_batch, chunks=chunks), dim=BATCH_DIM)


def cat_along_batch(data: Sequence[T]) -> T:
    return combine_tensors(data, bt.cat_along_batch)
```

A nested function has no logic. If one needs logic, the logic goes into a helper and is tested once.

### 3.3 The error contract (P8)

| Situation                                                                       | Exception         | Raised by         |
| ------------------------------------------------------------------------------- | ----------------- | ----------------- |
| A leaf is not a `torch.Tensor` (map, split, combine)                            | `TypeError`       | `utils.structure` |
| `combine` receives no item or structures with different layout                  | `ValueError`      | `combine_tensors` |
| Leaves disagree on the split, chunk or shuffle dimension                        | `ValueError`      | `get_size`        |
| `pad_size < 0`                                                                  | `ValueError`      | `tensor.padding`  |
| `permutation`/`index` has the wrong ndim or length                              | `ValueError`      | `tensor.indexing` |
| `lengths` is not a 1-D integer tensor                                           | `ValueError`      | `tensor.masking`  |
| `repeats < 0`, `chunks <= 0`, a 1-D tensor given to a seq function, a bad dtype | torch's own error | torch (unchanged) |

`batchtensor` validates only what torch cannot see or what torch accepts but means something else
(a negative pad size crops, a duplicated index is not a permutation). Everything else is left to
torch, so the messages stay the ones users already know.

## 4. Public API after the change

### 4.1 `tensor`

| Function(s)                                                                                            | Change                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sum`, `mean`, `prod`, `amax`, `amin`, `argmax`, `argmin`, `max`, `min`, `median` `_along_{batch,seq}` | `keepdim` becomes keyword-only: `f(tensor, *, keepdim=False)`. No other change (P12).                                                                                                                                                                                                    |
| `sort`, `argsort` `_along_{batch,seq}`                                                                 | `f(tensor, *, descending=False, stable=False)`. The untyped `**kwargs` is removed (F8).                                                                                                                                                                                                  |
| `cumsum`, `cumprod`                                                                                    | No change.                                                                                                                                                                                                                                                                               |
| `slice`, `select`, `chunk`, `split`                                                                    | No change. `step` becomes keyword-only: `slice_along_batch(t, start=0, stop=None, *, step=1)`.                                                                                                                                                                                           |
| `cat`, `stack`                                                                                         | No change.                                                                                                                                                                                                                                                                               |
| `repeat_along_seq`                                                                                     | No change. **Add `repeat_along_batch`** for symmetry (F11).                                                                                                                                                                                                                              |
| `index_select_along_{batch,seq}`                                                                       | Validation moves to `ValueError`: the index must be 1-D, or 2-D for `_seq` with `index.shape[0] == tensor.shape[0]`. The 2-D mode stays (D5).                                                                                                                                            |
| `permute_along_{batch,seq}`                                                                            | Kept (D6). Validation becomes `ValueError` for a non-1-D or wrong-length permutation. It does not check for duplicates (documented). `shuffle_*` calls it.                                                                                                                                                                                                                             |
| `shuffle_along_{batch,seq}`                                                                            | `f(tensor, *, generator=None)` where `generator` is `torch.Generator \| int \| None`. It calls `permute_along_*`. Uses `setup_torch_generator` (F9, F13).                                                                                                                                   |
| `pad_along_{batch,seq}`                                                                                | `f(tensor, pad_size, *, value=0.0, side="right")` with `side` in `{"left", "right"}`. A `value` that does not fit the dtype raises `ValueError` (F11).                                                                                                                                   |
| `lengths_to_mask`                                                                                      | `f(lengths, *, max_len=None)`. Rejects non-integer dtype and non-1-D input with `ValueError`. Negative lengths give an all-`False` row. `max_len < lengths.max()` truncates (documented).                                                                                                |
| **`pad_sequences`** (new)                                                                              | `pad_sequences(tensors, *, value=0.0, side="right") -> Tensor`. Pads a list of tensors that differ along dim 0 into one `(batch, seq, ...)` tensor. Returns the tensor only. Lengths come from `[t.shape[0] for t in tensors]`. It is the batch-building step next to `lengths_to_mask`. |

`tensor.__all__` goes from 50 to 68 names: two added in this table (`repeat_along_batch`,
`pad_sequences`) and 16 from the new operations below (D10). See Appendix B.

**New operations (D10).** Eight operations, each as `_along_batch` and `_along_seq`, each with its
`nested` twin, its parity entry and its tests: `std`, `var`, `any`, `all`, `logsumexp`, `flip`,
`roll`, `topk`. Options are keyword-only (`std`/`var`: `correction=1, keepdim=False`; `any`/`all`/
`logsumexp`: `keepdim=False`; `flip`: none; `roll`: `shifts`; `topk`: `k`, `largest=True`,
`sorted=True`). `topk` returns `torch.return_types.topk` like `max` (D7). Masked reductions
(`mean_along_seq(x, mask=...)`) are not part of this set and need their own design.

### 4.2 `nested`

| Function(s)                                                                                                                                                             | Change                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Every twin of a `tensor` function                                                                                                                                       | Same name and options as in 4.1 (P3), implemented with the three helpers. Annotated `T -> T` when the leaf stays a tensor (P13).                                                                                     |
| `max`, `min`, `median`, `sort` `_along_*`                                                                                                                               | Annotated `NestedTensor -> NestedTensor`. Each leaf becomes a `torch.return_types.*` named tuple (P5: the structure is kept, the leaf is replaced). Documented and tested (D7).                                      |
| `chunk`, `split` `_along_*`                                                                                                                                             | Return `tuple[NestedTensor, ...]`. Empty data and mismatched leaves raise `ValueError` (F3, F4).                                                                                                                     |
| `cat`, `stack` `_along_*`                                                                                                                                               | Take `Sequence[T]`, return `T`. An empty sequence raises `ValueError`. `OrderedDict` and named tuples keep their type (F2).                                                                                          |
| `combine`, `split`                                                                                                                                                      | **Moved** to `utils.structure` as `combine_tensors`, `split_tensors` (F5).                                                                                                                                           |
| `abs`, `clamp`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh` | **Removed** (F7, D1). Use `apply_to_tensors(batch, torch.exp)`. The user guide documents this idiom.                                                                                                                 |
| `clone`, `detach`, `contiguous`, `pin_memory`, `to`                                                                                                                     | Kept in `misc.py` (D2). `contiguous(data, *, memory_format=torch.contiguous_format)`. `to(data, *args, **kwargs)` stays: it mirrors the torch overloads. `pin_memory` keeps no argument.                             |
| `to_numpy`, `from_numpy`                                                                                                                                                | Kept. `to_numpy` detaches and moves to CPU (documented, P11). `from_numpy` shares memory with the arrays (documented). Both accept any leaf for conversion (the exception of P6) and use `recursive_apply` directly. |
| `as_tensor`                                                                                                                                                             | **Removed** (F6, D3). Use `recursive_apply(data, torch.as_tensor)` when a conversion of scalars is needed.                                                                                                           |
| `lengths_to_mask`, `pad_sequences`                                                                                                                                      | **No nested twin, by design.** Their input is not a batch of tensors (lengths, a list of tensors). Listed as the parity exceptions.                                                                                  |
| `types.NestedTensor`                                                                                                                                                    | Becomes public in `batchtensor.nested`. The alias excludes `str` and `bytes`: `Tensor \| Mapping[Any, NestedTensor] \| list[NestedTensor] \| tuple[NestedTensor, ...]` (F12).                                        |

The parity exceptions (`lengths_to_mask`, `pad_sequences`) are the only entries of the allow-list in
`test_api_parity`.

### 4.3 `utils` and `constants`

| Name                                               | Change                                                                                                |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `utils.seed.setup_torch_generator`                 | Unchanged. Now used by `shuffle_*` (F13).                                                             |
| `utils.structure.apply_to_tensors`                 | **New.** The extension point for any tensor function (also replaces the removed pointwise functions). |
| `utils.structure.split_tensors`, `combine_tensors` | **New** (moved from `nested`).                                                                        |
| `utils.structure.get_size`                         | **New.** Replaces the private `_get_size`.                                                            |
| `constants.BATCH_DIM`, `SEQ_DIM`                   | Unchanged.                                                                                            |

### 4.4 Size

The change removes `pointwise.py`, `trigo.py`, their tests, `as_tensor`, `nested.combine`
and `nested.split` (about 700 lines of source), and adds `structure.py` (about 120 lines), `pad_sequences` and the
eight new operations. The number of public names goes from 130 (`50 + 80`) to 143
(`68 + 75`, Appendix B).

## 5. Dependents

- **`coola`.** `batchtensor` depends on `coola.recursive.recursive_apply`, `coola.recursive.rebuild`
  and `coola.random`. The `rebuild_*` functions are public module members but are not in
  `coola.recursive.__all__`. Ask for them to be exported (or accept the module path) before the
  first release (decision D8). The pin in `pyproject.toml` is `coola>=2.0.0a2,<3.0`; the README
  table still says `>=1.0,<2.0`.
- **Users.** The breaking changes are: positional options, `as_tensor`, the pointwise
  and trigonometric functions, `nested.combine`/`nested.split`, `RuntimeError` to `ValueError` in
  `index_select_along_seq`, the error type for non-tensor leaves, the error for empty `cat`/`stack`.
  A migration page lists each one with the replacement (Step 9).

## 6. Why these choices

- **A strict leaf rule over a permissive one (P6).** Skipping non-tensor leaves would make
  `slice_along_batch({"x": tensor, "meta": ["a", "b", "c"]})` return a batch whose `meta` still has
  three items. That is a silent batch corruption, the worst failure for this package. A clear
  `TypeError` costs the user one line (`{k: v for k, v in batch.items() if ...}`).
- **No `dim` argument (P1).** The package exists because "dim 0 and dim 1" is a convention the
  user does not want to repeat. A `dim` argument gives torch back, with extra API.
- **Remove pointwise functions (D1).** They carry no batch meaning. They cost 590 lines and 21
  entries in the API reference, and any new torch function needs the same boilerplate.
  `apply_to_tensors(batch, torch.exp)` is the same length as `nested.exp(batch)` plus one import,
  and it works for every torch function, not for 21 of them.
- **Keep two levels (P3).** The `tensor` level is useful alone, and the `nested` level is a
  mechanical twin. The parity test makes the twin free to maintain.

## 7. Evidence

### 7.1 Probes (run against `main`)

| Probe                                                         | Result on `main`                                         |
| ------------------------------------------------------------- | -------------------------------------------------------- |
| `sum_along_batch({"a": t, "b": "s"})`                         | `TypeError` from torch                                   |
| `slice_along_batch({"a": t, "b": None})`                      | `TypeError: 'NoneType' object is not subscriptable`      |
| `chunk_along_batch({"a": t, "b": 1}, 2)`                      | `AttributeError: 'int' object has no attribute 'chunk'`  |
| `shuffle_along_batch({"a": t, "b": "s"})`                     | `AttributeError: 'str' object has no attribute 'shape'`  |
| `shuffle_along_seq({"a": torch.arange(3)})`                   | `IndexError: tuple index out of range`                   |
| `cat_along_batch([])`, `chunk_along_batch({}, 2)`             | `{}` and `()`                                            |
| `cat_along_batch([OrderedDict(a=t)] * 2)`                     | `dict` (type lost)                                       |
| `chunk_along_batch(namedtuple(...)(t, t), 2)[0]`              | plain `tuple` (type lost)                                |
| `chunk_along_batch({"a": 3 rows, "b": 2 rows}, 2)`            | no error, misaligned chunks                              |
| `mean_along_batch(defaultdict / OrderedDict / dataclass)`     | type kept, dataclass entered                             |
| `max_along_batch({"a": t})["a"]`                              | `torch.return_types.max` (kept)                          |
| `as_tensor({"a": [1, 2, 3]})`                                 | `{'a': [tensor(1), tensor(2), tensor(3)]}`               |
| `index_select_along_seq(x, float 2-D index)` (batch mismatch) | `RuntimeError`                                           |
| `index_select_along_seq(zeros(2,3,4), 3-D index)`             | `IndexError: shape mismatch`                             |
| `permute_along_batch(x, 2-D index)`                           | passes the length check, then `IndexError` from torch    |
| `permute_along_batch(x, tensor([0, 0, 0]))`                   | returns the first row three times                        |
| `lengths_to_mask(tensor([-1, 2]))`, `tensor([1.5, 2])`        | accepted                                                 |
| `lengths_to_mask(tensor([5, 2]), max_len=3)`                  | truncates silently                                       |
| `shuffle_along_batch(x, 0)`                                   | `AttributeError: 'int' object has no attribute 'device'` |
| `pad_along_batch(arange(3), 1, value=1.5)`                    | `tensor([0, 1, 2, 1])` (truncated)                       |
| `to_numpy({"a": x.requires_grad_()})`                         | works (detached)                                         |
| nesting depth 50 000 (`[[[...tensor...]]]`)                   | works (`coola` engine is not recursive)                  |
| `tensor`/`nested` signature parity over `tensor.__all__`      | exact for all 49 shared names                            |

### 7.2 Cost

macOS, Python 3.14, CPU, `torch` from `.venv`, a dict of 10 tensors of shape `(8, 4)`,
mean of 2 000 calls (a single run; the numbers show the order of magnitude, not a benchmark):

| Call                                                        | Time            |
| ----------------------------------------------------------- | --------------- |
| `{k: bt.sum_along_batch(v) for ...}` (manual)               | 12.8 us         |
| `bn.sum_along_batch(data)` (today)                          | 26.9 us         |
| `recursive_apply` with the `isinstance` guard               | 18.6 us         |
| `bn.sum_along_batch(tensor)` / `bt.sum_along_batch(tensor)` | 1.8 us / 1.1 us |

The guard adds no visible cost: the guarded call (18.6 us) was not slower than today's call (26.9 us)
in this run. One run cannot explain the gap, so treat the numbers as an order of magnitude. The gate in Section 9 re-measures before merging Step 3 and asks only for "no slower than `main`
by more than 20 %".

### 7.3 Typing probe

With `ty`, `T = TypeVar("T", bound=NestedTensor)` and `def f(data: T) -> T` gives
`Tensor -> Tensor`, `dict[str, Tensor] -> dict[str, Tensor]`, and rejects `int = f(tensor)`.
`f({"a": [t]})` is revealed as `dict[Any, list[Tensor]]`. So the `T -> T` annotation works with the
recursive alias on the checker in use.

## 8. Migration steps

One branch and one PR per step, stacked when a step depends on the previous one (`CLAUDE.md`).
Each PR says what is changed, what is not, and the behaviour change. Each updates the changelog
entry or the tracking table with the PR number.

| Step | Branch                          | Content                                                                                                                                                                                                                                             | Breaking | PR |
| ---- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | --- |
| 1    | `docs/fix-contradictions`       | Fix Appendix C (README table, "returned unchanged", Pattern 3, "flat dict", the `suggestions.md` status). Delete `review/suggestions.md` or mark it superseded.                                                                          | No | #700 |
| 2    | `chore/enforce-layering`        | Add the `import-linter` contract (P14). Keep `pyright` and `ty` (D9).                                                                                                                                       | No | #701 |
| 3    | `refactor/structure-helpers`    | Add `utils/structure.py` with `apply_to_tensors`, `split_tensors`, `combine_tensors`, `get_size`. Move `nested.combine`/`split`/`_get_size` there. Switch every nested function to the helpers. Add `T -> T` typing and make `NestedTensor` public. | Yes (error types, container types, removed `nested.combine`/`split`) | #703 |
| 4    | `refactor/keyword-only-options` | Make options keyword-only. Replace `**kwargs` in `sort`/`argsort` by `stable`. Add the signature test.                                                                                                                                              | Yes | #704 |
| 5    | `refactor/remove-pointwise`     | Remove `pointwise.py`, `trigo.py`, `as_tensor` and their tests. Add the `apply_to_tensors` idiom to the user guide.                                                                                                                                 | Yes | #705 |
| 6    | `refactor/shuffle-seed`         | `shuffle_*` accepts an `int` seed through `setup_torch_generator` and keeps calling `permute_*`.                                                                                                  | Yes | #706 |
| 7    | `bugfix/validation-contract`    | Apply the error table of Section 3.3: `ValueError` in `index_select_along_*`, `lengths_to_mask`, `pad_*`. Regression test for each, checked to fail without the fix.                                                                                | Yes (`RuntimeError` to `ValueError`) | #707 |
| 8    | `feat/pad-side-and-sequences`   | Add `side` to `pad_along_*`, `repeat_along_batch`, `pad_sequences`.                                                                                                                                                                                 | No | #708 |
| 9    | `docs/target-state`             | Rewrite `design_principles.md` from the property table. Update `architecture.md`, the user guides and `package_checks.py`. Add a migration page that lists each break and its replacement.                                                          | No | #710 |
| 10   | `test/api-parity`               | Add `test_api_parity`, the differential tests, the autograd test and the device smoke test (Section 9). Can land early: it needs only the allow-list.                                                                                               | No | #709 |
| 11   | `feat/add-<operation>` (x8)     | One PR per operation of D10: `std`, `var`, `any`, `all`, `logsumexp`, `flip`, `roll`, `topk`. Each adds the `tensor` pair, the `nested` pair, tests, docstring examples and the user guide entry. | No | |

Steps 4 to 8 and 11 depend on 3 and are independent of each other after it. Step 10 can be the first PR
(it only fails if parity is already broken, which it is not).

## 9. Tests and gates

**Unit tests** (mirror the tree, `tests/unit/<package>/test_<module>.py`):

- `utils/test_structure.py`: for each helper, a typical dict, a list, a tuple, a named tuple, an
  `OrderedDict`, a `defaultdict`, a dataclass (map only), a nested mix, an empty container, a single
  leaf, a non-tensor leaf (`TypeError` with `match`), no input mutation, a deterministic repeat call,
  the whole output asserted against a literal. For `combine_tensors`: empty items, different keys
  (`ValueError`), different lengths, different key order. For `split_tensors`: `n == 0`, leaves with
  different part counts. For `get_size`: no leaf, disagreeing leaves, 0-d leaf.
- `tensor/*`: the new validations, `side="left"`, `pad_sequences` (ragged, equal lengths, empty list,
  1-D and 3-D tensors, dtype kept), `repeat_along_batch`, `lengths_to_mask` (float, negative,
  truncation, empty), `shuffle_*` with an `int` seed and a `Generator` (same seed, same result).
- `nested/*`: one parametrized case per function for a dict, a list and a tuple, using the shared
  fixture below.
- Remove `test_types.py` (it asserts nothing about the alias, `CLAUDE.md` forbids trivial structure
  tests). Replace it with the typing test below.

**Integration tests** (`tests/integration/`, no patching):

- `test_api_parity.py`: for every name in `tensor.__all__` that is not in the allow-list
  (`lengths_to_mask`, `pad_sequences`), the name exists in `nested.__all__`, and the parameters,
  defaults, kinds and annotations (after `T -> T`) match. For every name in `nested.__all__`, the name
  exists in `tensor.__all__` or in the explicit nested-only list (`clone`, `detach`, `contiguous`,
  `pin_memory`, `to`, `to_numpy`, `from_numpy`). This is the gate of P3.
- `test_differential.py`: one table of `(tensor_fn, nested_fn, kwargs)`. On a single tensor,
  `nested_fn(t) == tensor_fn(t)`. On `{"a": t, "b": [t, t]}`, every leaf equals the tensor result and
  the structure is the same. This replaces the repeated per-function checks in the nested unit tests.
- `test_torch_reference.py`: a table of `(tensor_fn, torch_fn)` that checks P2 against torch
  (`sum_along_batch(t) == torch.sum(t, dim=0)`, ...), for float, int and bool dtypes where torch allows.
- `test_autograd.py`: gradients flow through a nested chain (`slice`, `mean`, `cumsum`) (P11).
- `test_device.py`: skipped when no accelerator is present. Runs the `shuffle`, `index_select` and
  `lengths_to_mask` paths on `cuda` or `mps` (the F3 bug class of `suggestions.md`).
- `test_typing.py`: a `typing.assert_type` file checked by `ty` for `Tensor`, `dict`, `list`, `tuple`.
- `test_pipeline.py`: keep, extended with `pad_sequences` + `lengths_to_mask` + `shuffle` + `chunk`
  on a mixed nested batch. Also covers every patched behaviour with the real collaborator.

**Gates (CI):**

- `ruff check`, `ruff format`, `ty check`, `lint-imports`.
- Unit and integration tests on the oldest supported `torch` (`2.4`) and the latest.
- 100 % coverage of `src` with no `pragma: no cover` that could be tested.
- Doctest job on every docstring.
- Benchmark: Section 7.2 script must not regress by more than 20 % against `main`.
- Regression test for every bug fixed in Steps 3, 6 and 7, verified to fail without the fix
  (`git stash push src`).

## 10. Decisions for the maintainer

| #   | Question                                                                                                  | Recommendation                                                                                                                                                                                                                               |
| --- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Remove the 21 pointwise and trigonometric functions in favour of `apply_to_tensors`?                      | **Yes.** They are not batch functions (P3 fails), and `apply_to_tensors` covers every torch function. Alternative: keep them as they are and accept the 590 lines, the builtin shadowing and a `tensor` level without twins.                 |
| D2  | Keep `clone`, `detach`, `contiguous`, `pin_memory`, `to` in `nested`?                                     | **Yes.** They are batch lifecycle operations (`batch = to(batch, "cuda")` is the most common call on a nested batch). They stay as the nested-only list in the parity test.                                                                  |
| D3  | Remove `as_tensor`?                                                                                       | **Yes.** It is ill-defined on lists (F6). `recursive_apply(data, torch.as_tensor)` is the same length and explicit about its leaf semantics. Alternative: keep and document that lists are containers.                                       |
| D4  | Support dataclasses in `combine_tensors` and `split_tensors`?                                             | **Not now.** Map functions already enter dataclasses through `coola`. The parallel walk needs a rebuild rule for dataclasses; add it when a user asks (or when `coola` exposes a multi-input apply). Document the difference.                |
| D5  | Keep the 2-D mode of `index_select_along_seq`?                                                            | **Yes**, validated and documented. It is the per-sample gather users need on `(batch, seq, ...)`. Alternative: a separate `gather_along_seq`, with a 1-D-only `index_select_along_seq`. Revisit if a third mode appears.                     |
| D6  | Remove `permute_along_*`? | **Decided: keep.** The name and the length check document the intent. It is validated (`ValueError`), documented as not checking duplicates, and the docstring warns about `torch.permute`. `shuffle_*` keeps calling it. |
| D7  | Keep named-tuple leaves for `max`, `min`, `median`, `sort` in `nested`?                                   | **Yes.** It follows P5 (leaf replaced, structure kept). Alternative: a named tuple of structures `(values=tree, indices=tree)`. It is handier but breaks P5 and needs its own helper.                                                        |
| D8  | Use `coola.recursive.rebuild` from `batchtensor`?                                                         | **Yes**, after asking `coola` to export `rebuild_mapping` and `rebuild_sequence` from `coola.recursive`. Alternative: a local copy (about 20 lines) that diverges from `coola`.                                                              |
| D9  | Drop `pyright` and keep `ty` only? | **Decided: keep both.** Both checkers run in CI and both must pass. The `T -> T` typing (P13) must be checked with both (a `pyright` run of `test_typing.py`). |
| D10 | Add more operations (`std`, `var`, `any`, `all`, `logsumexp`, `topk`, `flip`, `roll`, masked reductions)? | **Decided: add.** Eight operations are in the target (Section 4.1, Step 11). Rule: one PR per operation with its `tensor` function, its `nested` twin, its parity entry and its tests. Masked reductions need their own design and are not in the set. |

Decisions changed by the maintainer: D6 (keep `permute_along_*`), D9 (keep both checkers), D10 (add more operations).

## 11. Not covered

- Masked reductions and operations beyond the eight of D10.
- Sequence-first or channel-first layouts, a configurable `seq_dim`.
- Dataclass support in `combine_tensors` and `split_tensors` (D4).
- Reporting the path of a bad leaf in the `TypeError` (needs a path-aware `coola` engine).
- Sparse, nested (jagged) or masked `torch` tensors, and `numpy`/`pandas` leaves.
- A `torch.compile` audit of the wrappers.
- Release engineering (the version table, the classifiers, the nightly jobs).

## Appendix A. Status of `review/suggestions.md`

| Item (section.number)                        | Status on `main`                                                                    | In target    |
| -------------------------------------------- | ----------------------------------------------------------------------------------- | ------------ |
| 1.1 `repeat_along_seq` hard-coded `sizes[1]` | Fixed (uses `SEQ_DIM`, raises `ValueError`)                                         | Kept         |
| 1.2 `permute` uses `shape[0]`/`shape[1]`     | Fixed (uses the constants, `ValueError`)                                            | Kept (D6)    |
| 1.3 `shuffle` device mismatch                | Fixed (permutation on the generator device, then moved)                             | Kept         |
| 1.4 nested `shuffle` on empty data / sizes   | Fixed (`_get_size` handles empty and checks sizes)                                  | `get_size`   |
| 1.5 `index_select_along_seq` 2-D branch      | Fixed (`reshape`, `.long()`, batch check). Error type and 3-D index still open (F9) | Step 7       |
| 1.6 `to_numpy` grad / CUDA                   | Fixed (`detach().cpu()`)                                                            | Kept         |
| 1.7 nested joining/slicing flat dict only    | Fixed (recursive). Container types still lost (F2)                                  | Step 3       |
| 1.8 named-tuple results                      | Fixed in behaviour and docs. Needs a pinned test and a decision (D7)                | Step 3       |
| 1.9 `lengths_to_mask` validation             | Partly (ndim). Dtype, negative values, truncation still open (F9)                   | Step 7       |
| 1.10 no nested `lengths_to_mask`             | Open. Decided: by design, listed as an exception                                    | Step 10      |
| 2.1 loose return types                       | Open (F12)                                                                          | Step 3       |
| 2.2 `NestedTensor` matches `str`             | Open (F12)                                                                          | Step 3       |
| 2.3 hard-coded dims                          | By design (P1), documented in `design_principles.md`                                | Kept         |
| 2.4 padding options                          | Open (F11)                                                                          | Step 8       |
| 2.5 missing operations                       | Open. Eight operations are added (D10); masked reductions stay open               | Step 11      |
| 2.6 builtin shadowing                        | Resolved by D1                                                                      | Step 5       |
| 2.7 `to`/`contiguous`/`pin_memory`           | Kept as pass-through (D2)                                                           | Step 4       |
| 2.8 argument validation                      | Open (F9)                                                                           | Step 7       |
| 3.1 to 3.3 duplication                       | Done for `combine`/`split`. Remaining one-liners are intentional                    | Kept         |
| 3.4 large reduction modules                  | Accepted: the size is docstring, which is the product                               | Kept         |
| 3.5 parametrized differential tests          | Open (F16)                                                                          | Step 10      |
| 4.1 empty `tests/unit/recursive`             | Gone                                                                                | n/a          |
| 4.2 integration coverage                     | Open                                                                                | Step 10      |
| 4.3 property-based tests                     | Not adopted: the differential tests give the same coverage without a new dependency | Deferred     |
| 5.1 `known-first-party`                      | Fixed (`batchtensor`)                                                               | n/a          |
| 5.2 to 5.6 ruff, classifiers, log level      | Fixed (`C90` selected, `INFO`, no license classifier)                               | n/a          |
| 5.7 numpy documentation                      | Documented in `design_principles.md`                                                | Kept         |
| 6.x documentation                            | `design_principles.md` exists. Contradictions remain (Appendix C)                   | Step 1, 9    |

## Appendix B. API inventory

| Family                         | `tensor` now | `tensor` target | `nested` now | `nested` target |
| ------------------------------ | ------------ | --------------- | ------------ | --------------- |
| reduction (10 ops x 2 dims)    | 20           | 20              | 20           | 20              |
| comparison (`sort`, `argsort`) | 4            | 4               | 4            | 4               |
| math (`cumsum`, `cumprod`)     | 4            | 4               | 4            | 4               |
| slicing (4 ops x 2 dims)       | 8            | 8               | 8            | 8               |
| joining (`cat`, `stack`)       | 4            | 4               | 4            | 4               |
| `repeat`                       | 1            | 2               | 1            | 2               |
| indexing                       | 2            | 2               | 2            | 2               |
| permutation (`permute`)        | 2            | 2               | 2            | 2               |
| shuffle                        | 2            | 2               | 2            | 2               |
| padding                        | 2            | 2               | 2            | 2               |
| masking and sequences          | 1            | 2               | 0            | 0               |
| pointwise and trigonometric    | 0            | 0               | 21           | 0               |
| new operations (D10)           | 0            | 16              | 0            | 16              |
| misc (lifecycle)               | 0            | 0               | 5            | 5               |
| conversion                     | 0            | 0               | 3            | 2               |
| `combine`, `split`             | 0            | 0               | 2            | 0               |
| **Total**                      | **50**       | **68**          | **80**       | **75**          |

(`nested` now: 80 names in `__all__`; target: 75. `utils` gains four public helpers.)

## Appendix C. Documentation fixes (Step 1 and Step 9)

| File                             | Problem                                                                                                                                                                 | Fix                                                                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`                      | Version table says `coola >=1.0,<2.0` and `numpy >=1.24,<2.0` for `main`. `pyproject.toml` says `coola>=2.0.0a2,<3.0` and `numpy>=1.24,<3.0`.                           | Update the `main` row from `pyproject.toml`.                                                                                       |
| `docs/docs/design_principles.md` | "Non-tensor leaves are returned unchanged" is false (F1). "Joining and slicing accept a flat dict" is false (they recurse).                                             | Rewrite from the property table (P6, P5).                                                                                          |
| `docs/docs/architecture.md`      | Mentions `pyright`. Pattern 3 shows the old flat-dict `chunk_along_batch`. "Adding New Data Types" lists "custom classes with appropriate handlers" without saying how. | Use `ty`. Replace Pattern 3 with the helpers of Section 3. Point to `coola.recursive.register_transformers` for custom containers. |
| `docs/docs/architecture.md`      | Lists seven tensor sub-modules; `masking.py` and `padding.py` are missing from the list.                                                                                | Regenerate the module list from the tree.                                                                                          |
| `docs/docs/uguide/nested.md`     | "Supported Structures" omits dataclasses, `OrderedDict`, named tuples. No statement about non-tensor leaves.                                                            | Add a "Leaves and errors" section from P5, P6 and Section 3.3. Add the `apply_to_tensors` idiom.                                   |
| `src/batchtensor/__init__.py`    | Claims "Type safety". True only after Step 3.                                                                                                                           | Keep the claim, land it with Step 3.                                                                                               |
| `review/suggestions.md`          | Stale (Appendix A).                                                                                                                                                     | Delete it in Step 1 and keep this file as the reference.                                                                           |
