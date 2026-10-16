# Codebase review: suggestions

Scope: `src/batchtensor` (about 5.9k lines, most of it docstrings), the test layout, `pyproject.toml`, CI and docs structure.
I found these by reading the code. I did not run any of them, so check the "likely bug" items with a quick repro before fixing.

## 1. Likely bugs and edge cases

| #   | Location                                                                            | Issue                                                                                                                                                                                                                                                                                                                                                     | Suggested fix                                                                                                                                       |
| --- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `tensor/joining.py` `repeat_along_seq`                                              | Hard-codes `sizes[1]` instead of `SEQ_DIM`. A 1-D tensor raises a bare `IndexError`.                                                                                                                                                                                                                                                                      | Use `SEQ_DIM`. Raise a clear `ValueError` when `tensor.dim() < 2`.                                                                                  |
| 2   | `tensor/permutation.py` `permute_along_*`                                           | Uses `shape[0]` and `shape[1]` instead of `BATCH_DIM` and `SEQ_DIM`. It raises `RuntimeError` for what is an argument error.                                                                                                                                                                                                                              | Use the constants. Raise `ValueError`, or keep `RuntimeError` and document it.                                                                      |
| 3   | `tensor/permutation.py` `shuffle_along_*`                                           | `torch.randperm(..., generator=generator)` creates the permutation on the generator's device (CPU by default). With a CUDA tensor, `index_select` then fails on a device mismatch.                                                                                                                                                                        | Move the permutation to `tensor.device`. Or create it with `device=generator.device` and then call `.to(tensor.device)`.                            |
| 4   | `nested/permutation.py` `shuffle_along_*`                                           | `next(dfs_iterate(data))` raises `StopIteration` on empty data. It also reads the size from the first tensor only, so tensors with different batch sizes fail later with an obscure error.                                                                                                                                                                | Handle empty input explicitly. Validate that all tensors share the batch size.                                                                      |
| 5   | `tensor/indexing.py` `index_select_along_seq`                                       | The 2-D index branch calls `.view(...)`, which fails on non-contiguous results. Only that branch applies `.long()`. Nothing checks `index.shape[0] == tensor.shape[0]`.                                                                                                                                                                                   | Use `.reshape` or `tensor.gather`. Validate shapes. Cast in both branches or document the dtype requirement.                                        |
| 6   | `nested/conversion.py` `to_numpy`                                                   | `tensor.numpy()` fails for tensors that require grad or live on CUDA.                                                                                                                                                                                                                                                                                     | Use `tensor.detach().cpu().numpy()`. Or document the limitation and test it.                                                                        |
| 7   | `nested/joining.py`, `nested/slicing.py`                                            | `cat_*`, `stack_*`, `chunk_*` and `split_*` accept only flat `dict[Hashable, Tensor]`. This contradicts the package's "nested" promise: lists, tuples and deeper dicts are not handled. Empty `chunk_*` and `split_*` input silently returns `()`. `type(item)(...)` also breaks for dict subclasses with a different constructor, such as `defaultdict`. | Implement the recursion, for example by zipping nested structures with `coola` helpers. Or narrow the docs and type names to say "dict of tensors". |
| 8   | `nested/reduction.py` `max_*`, `min_*`, `median_*`; `nested/comparison.py` `sort_*` | The tensor versions return `torch.return_types.*` namedtuples. It is unclear whether `recursive_apply` keeps them as namedtuples or turns them into plain tuples or lists.                                                                                                                                                                                | Add a test that pins the behaviour. Document it in the docstrings.                                                                                  |
| 9   | `tensor/masking.py` `lengths_to_mask`                                               | `lengths.max().item()` forces a GPU sync. There is no validation for negative lengths, non-1-D input or `max_len < lengths.max()` (which silently truncates).                                                                                                                                                                                             | Document the sync. Validate `lengths.ndim == 1` and `max_len`.                                                                                      |
| 10  | `nested/__init__.py`                                                                | `lengths_to_mask` has no nested counterpart, so the two APIs are not symmetric.                                                                                                                                                                                                                                                                           | Decide whether `masking` should be part of the nested module. Document the exception if not.                                                        |

## 2. API and design

1. **Return types are too loose.** Every nested function returns `NestedTensor`, so `nested.sum_along_batch(tensor)` loses the fact that a tensor goes in and a tensor comes out. Add a `TypeVar` or `@overload` for the `Tensor -> Tensor` and `dict -> dict` cases. Type checkers then give precise results.
2. **`NestedTensor` alias.** `Sequence['NestedTensor']` also matches `str` and `bytes`. Exclude them, or document the behaviour. Consider making the alias public in `batchtensor.nested`, if it is not exported already.
3. **Hard-coded batch=0 and seq=1.** Everything is fixed to those two dims. A generic helper such as `*_along_dim(tensor, dim=...)`, or a configurable `seq_dim`, would cover batch-first versus sequence-first layouts. At least document the limitation prominently, since many users have `(seq, batch, ...)` data.
4. **Padding.** `pad_along_*` only pads at the end with a constant. Add `side="left" | "right"` and a way to pad to a target length (`pad_to_length`). Padding a list of variable-length tensors into one batch (`pad_sequence`-like) is the most common neighbouring need.
5. **Missing common operations.** Candidates:
   - `squeeze`, `unsqueeze`, `flip`, `roll`
   - `topk`, `gather`
   - `std`, `var`, `logsumexp`, `any`, `all`
   - `sqrt`, `pow`, `floor`, `ceil`, `round`, `sigmoid`
   - masked reductions (`mean_along_seq(x, mask=...)`), which are the main reason to have a seq dimension
   - batch-size helpers (`get_batch_size(data)`, `check_same_batch_size(data)`)
6. **Pointwise names.** `nested.abs` and `nested.clamp(min, max)` shadow builtins. The ruff `A` rule is enabled, so they are probably suppressed or ignored somewhere. Consider documenting the intended import style (`import batchtensor.nested as bn`).
7. **`nested.to`, `contiguous`, `pin_memory`.** Add a `to_device` alias, and `non_blocking` and `memory_format` pass-through, if they are not already covered by `*args, **kwargs`.
8. **Argument validation.** Several functions take ints (`chunks`, `repeats`, `pad_size`) and validate only some of them (`_pad` checks, `repeat_along_seq` does not). Make validation uniform.

## 3. Duplication and maintainability

1. The `*_along_batch` and `*_along_seq` pairs are 2 x N near-identical one-liners in `tensor/`, and 2 x N again in `nested/`. The docstrings are the valuable part, so keep them. Reduce the code with small private helpers (`_reduce(tensor, fn, dim, keepdim)`, `_apply(data, fn, **kw)`).
2. In `nested/joining.py`, `cat_*` and `stack_*` repeat the same empty check, key check and dict rebuild four times. One `_apply_to_dicts(data, fn)` helper replaces them. The same applies to `chunk_*` and `split_*` in `nested/slicing.py`.
3. `nested/math.py` uses `partial(bt.cumprod_along_batch)` with no arguments. The `partial` is redundant.
4. `nested/reduction.py` is 816 lines and `tensor/reduction.py` is 695, almost all docstring. Adding `std`, `var` and similar would make them larger still. Consider splitting into `reduction_minmax.py`, `reduction_stats.py`, or similar. Alternatively, accept the size and add a doc-generation check so docstring examples cannot drift.
5. Parametrised tests that iterate over every `(function, torch_reference)` pair would make adding a new op a one-line change. Check whether the current unit tests already do this.

## 4. Tests

1. `tests/unit/recursive/` is an empty directory with no `__init__.py`. Delete it, or add the tests that were meant to go there.
2. The integration tests cover only `nested/test_conversion.py`. Add:
   - a CUDA or MPS smoke test, skipped when unavailable
   - an autograd test (gradient flows through nested ops)
   - mixed dict, list and tuple nesting for every nested function
   - empty containers, 0-size batches, 1-D tensors and non-contiguous tensors
   - dtype coverage (int, bool, half and bfloat16 for reductions and sorts)
3. Property-based tests (`hypothesis`) would cover the invariants cheaply:
   - `permute(permute(x, p), argsort(p)) == x`
   - `cat(split(x)) == x`
   - `slice` agrees with Python slicing
4. The `ci-doctest` workflow already exists. Make sure the docstring examples for nested functions use deterministic seeds so they stay stable across torch versions.
5. `tests/package_checks.py` is not part of the normal unit tests. Check that it still exercises a representative sample of the public API after the 3.11 modernisation.

## 5. Packaging and tooling

1. `pyproject.toml` has `[tool.ruff.lint.isort] known-first-party = ["src"]`. The package name is `batchtensor`, so this should almost certainly be `["batchtensor"]`.
2. `[tool.ruff.lint.mccabe] max-complexity = 10` has no effect because the `C90` rule is not selected. Either add `"C90"` or remove the section.
3. `lint.fixable` repeats the full `select` list, and `lint.unfixable = []` is the default. Remove both. Ruff's default already allows fixes for all selected rules.
4. The classifier `License :: OSI Approved :: BSD License` is redundant (and deprecated) next to `license = "BSD-3-Clause"` plus `license-files`. PEP 639 says to drop the classifier. Check whether hatchling warns.
5. The classifiers list Python 3.15. Check that CI really covers it. Also check whether "Development Status :: 4 - Beta" still matches a `0.1.2a1` version.
6. `tool.pytest.ini_options` has `log_level = "DEBUG"`. This makes failure output noisy. `INFO` is usually enough.
7. The `numpy` extra is optional, but `nested/conversion.py` has a `from_numpy` / `to_numpy` pair. State in the docstrings and docs that numpy is needed only for the optional pieces. Consider a friendlier error when it is missing.
8. `torch>=2.4,<3.0` and `coola>=1.0,<2.0` are reasonable. The test matrix (`ci-test-deps`, `nightly-test-package-dep`) is already there, so confirm it covers the oldest supported torch.
9. The `dev/generate_versions.py` and `dev/config/package_versions.json` workflow is a bot-driven one. Document it in `CONTRIBUTING.md`, so contributors know not to edit the JSON by hand.

## 6. Docs

1. `docs/docs/architecture.md` exists. Add a short "design principles" page covering the conventions: batch is dim 0, seq is dim 1, nested functions apply the tensor function through `recursive_apply`, and what happens with non-tensor leaves.
2. Document the semantics that are not obvious from names:
   - which functions return views and which return copies (`slice`, `select`, `chunk`, `split` are views and share memory)
   - what `nested` does with non-tensor leaves
   - how namedtuple results (`max`, `min`, `median`, `sort`) are represented in nested output
3. Add a short "common recipes" page: padding and masking a batch of variable-length sequences, shuffling a dataset dict, and splitting a nested batch into micro-batches.
4. The top-level `batchtensor/__init__.py` docstring says "Type safety" and "Comprehensive documentation". Once the overloads in section 2 land, the type-safety claim becomes true. Until then it is generous for functions that return `NestedTensor`.

## 7. Suggested order of work

1. **Quick wins (small PRs):** items 1, 2 and 6 in section 1, items 1 to 3 in section 5, and the empty `tests/unit/recursive` directory.
2. **Correctness:** items 3, 4, 5 and 8 in section 1, with tests for each.
3. **Nested joining and slicing (item 7 in section 1).** Decide between true nested support and narrowing the contract.
4. **Typing overloads** (section 2, item 1).
5. **New features** (padding options, masked reductions, missing ops), one small PR each.
