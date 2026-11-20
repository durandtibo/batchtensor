# Home

<p align="center">
    <a href="https://github.com/durandtibo/batchtensor/actions/workflows/ci.yaml">
        <img alt="CI" src="https://github.com/durandtibo/batchtensor/actions/workflows/ci.yaml/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/batchtensor/actions/workflows/nightly-tests.yaml">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/batchtensor/actions/workflows/nightly-tests.yaml/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/batchtensor/actions/workflows/nightly-package.yaml">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/batchtensor/actions/workflows/nightly-package.yaml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/durandtibo/batchtensor">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/batchtensor/branch/main/graph/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/batchtensor/">
        <img alt="Documentation" src="https://github.com/durandtibo/batchtensor/actions/workflows/docs.yaml/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/batchtensor/dev/">
        <img alt="Documentation" src="https://github.com/durandtibo/batchtensor/actions/workflows/docs-dev.yaml/badge.svg">
    </a>
    <br/>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"
             alt="Ruff"
             style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/batchtensor/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/batchtensor">
    </a>
    <a href="https://pypi.org/project/batchtensor/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/batchtensor.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/batchtensor">
    </a>
    <br/>
    <a href="https://pepy.tech/project/batchtensor">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/batchtensor">
    </a>
    <a href="https://pepy.tech/project/batchtensor">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/batchtensor/month">
    </a>
    <br/>
</p>

## Overview

`batchtensor` is a lightweight library built on top of [PyTorch](https://pytorch.org/) to manipulate
batches of tensors, and nested data structures (dictionaries, lists and tuples) of such batches.

It assumes one data layout:

- the **batch** dimension is the first dimension (dimension 0);
- the **sequence** dimension is the second dimension (dimension 1).

A function name says on which dimension it works: `sum_along_batch` reduces dimension 0 and
`slice_along_seq` slices dimension 1. No function takes a `dim` argument.

## Key Features

- **Two levels with the same API**: `batchtensor.tensor` works on one tensor and
  `batchtensor.nested` works on a dict, list or tuple of tensors. A function has the same name and
  options at both levels.
- **Structure preserving**: a nested function returns the structure it received, with the same
  container types and keys.
- **Thin over PyTorch**: a `tensor` function is one torch call with the dimension fixed. It does
  not copy, cast or move data on its own, and autograd works.
- **Strict on inputs**: a leaf that is not a tensor raises a `TypeError`; arguments that torch would
  accept with another meaning are validated.
- **Padded batches**: `pad_sequences`, `lengths_to_mask` and the masked reductions handle batches of
  sequences of different lengths.
- **Typed and documented**: every function has type hints and a docstring with a runnable example.
- **Lightweight**: the only required dependencies are PyTorch and
  [coola](https://github.com/durandtibo/coola).

## Main Modules

- **[batchtensor.tensor](refs/tensor.md)**: operations on individual tensors
- **[batchtensor.nested](refs/nested.md)**: the same operations on nested data structures, plus
  `to`, `clone`, `detach`, `batch_size`, `seq_len` and NumPy conversion
- **[batchtensor.utils](refs/utils.md)**: the helpers that traverse nested structures
  (`apply_to_tensors`, `split_tensors`, `combine_tensors`, `get_size`), the `NestedTensor` type and
  random-permutation helpers
- **[batchtensor.constants](refs/constants.md)**: `BATCH_DIM` and `SEQ_DIM`

## Motivation

Let's imagine you have a batch which is represented by a dictionary with three tensors, and you want
to take the first 2 items.
`batchtensor` provides the function `slice_along_batch` that allows slicing all the tensors:

```pycon
>>> import torch
>>> from batchtensor.nested import slice_along_batch
>>> batch = {
...     "a": torch.tensor([[2, 6], [0, 3], [4, 9], [8, 1], [5, 7]]),
...     "b": torch.tensor([4, 3, 2, 1, 0]),
...     "c": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
... }
>>> slice_along_batch(batch, stop=2)
{'a': tensor([[2, 6], [0, 3]]), 'b': tensor([4, 3]), 'c': tensor([1., 2.])}

```

Similarly, it is possible to split a batch into multiple batches by using the
function `split_along_batch`:

```pycon
>>> import torch
>>> from batchtensor.nested import split_along_batch
>>> batch = {
...     "a": torch.tensor([[2, 6], [0, 3], [4, 9], [8, 1], [5, 7]]),
...     "b": torch.tensor([4, 3, 2, 1, 0]),
...     "c": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
... }
>>> split_along_batch(batch, split_size_or_sections=2)
({'a': tensor([[2, 6], [0, 3]]), 'b': tensor([4, 3]), 'c': tensor([1., 2.])},
 {'a': tensor([[4, 9], [8, 1]]), 'b': tensor([2, 1]), 'c': tensor([3., 4.])},
 {'a': tensor([[5, 7]]), 'b': tensor([0]), 'c': tensor([5.])})

```

Please check the [user guide](uguide/nested.md) and [API reference](refs/nested.md) to see all the
implemented functions and detailed examples.

## Where to go next

- **[Get Started](get_started.md)**: installation instructions
- **[User Guide](uguide/tensor.md)**: tutorials and examples
- **[Design principles](design_principles.md)**: how the library behaves, and the trade-offs of
  each choice
- **[Architecture](architecture.md)**: how the library is built
- **[API Reference](refs/nested.md)**: complete function documentation
- **[GitHub Repository](https://github.com/durandtibo/batchtensor)**: source code and issue tracker

## API stability

:warning: While `batchtensor` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `batchtensor` to a new version will possibly break any code
that was using the old version of `batchtensor`.

## License

`batchtensor` is licensed under BSD 3-Clause "New" or "Revised" license available
in the [LICENSE](https://github.com/durandtibo/batchtensor/blob/main/LICENSE) file.
