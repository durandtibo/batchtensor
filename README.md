# batchtensor

<p align="center">
    <a href="https://github.com/durandtibo/batchtensor/actions/workflows/ci.yaml">
        <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/durandtibo/batchtensor/ci.yaml?branch=main&label=CI&logo=github"></a>
    <a href="https://github.com/durandtibo/batchtensor/actions/workflows/nightly-package.yaml">
        <img alt="Nightly Package Tests" src="https://img.shields.io/github/actions/workflow/status/durandtibo/batchtensor/nightly-package.yaml?label=nightly&logo=github"></a>
    <a href="https://codecov.io/gh/durandtibo/batchtensor">
        <img alt="Codecov" src="https://img.shields.io/codecov/c/github/durandtibo/batchtensor?logo=codecov"></a>
    <a href="https://durandtibo.github.io/batchtensor/">
        <img alt="Documentation (stable)" src="https://img.shields.io/badge/docs-stable-blue?logo=readthedocs&logoColor=white"></a>
    <br/>
    <a href="https://pypi.org/project/batchtensor/">
        <img alt="PyPI version" src="https://img.shields.io/pypi/v/batchtensor?logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/batchtensor/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/batchtensor?logo=python&logoColor=white"></a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/batchtensor"></a>
    <br/>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img alt="Doc style: google" src="https://img.shields.io/badge/docstyle-google-3666d6"></a>
    <a href="https://scorecard.dev/viewer/?uri=github.com/durandtibo/batchtensor">
        <img alt="OpenSSF Scorecard" src="https://img.shields.io/ossf-scorecard/github.com/durandtibo/batchtensor?label=openssf%20scorecard"></a>
    <br/>
    <a href="https://pepy.tech/project/batchtensor">
        <img alt="Downloads" src="https://img.shields.io/pepy/dt/batchtensor"></a>
    <a href="https://pepy.tech/project/batchtensor">
        <img alt="Monthly downloads" src="https://img.shields.io/pepy/dm/batchtensor"></a>
    <a href="https://github.com/durandtibo/batchtensor/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/durandtibo/batchtensor?logo=github"></a>
</p>

## Overview

`batchtensor` is a lightweight library built on top of [PyTorch](https://pytorch.org/) to manipulate
nested data structures with PyTorch tensors.
This library provides functions for tensors where the first dimension is the batch dimension.
It also provides functions for tensors representing a batch of sequences where the first dimension
is the batch dimension and the second dimension is the sequence dimension.

- [Motivation](#motivation)
- [Documentation](https://durandtibo.github.io/batchtensor/)
- [Installation](#installation)
- [Contributing](#contributing)
- [API stability](#api-stability)
- [License](#license)

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

Please check the documentation to see all the implemented functions.

## Documentation

- [latest (stable)](https://durandtibo.github.io/batchtensor/): documentation from the latest stable
  release.
- [main (unstable)](https://durandtibo.github.io/batchtensor/main/): documentation associated to the
  main branch of the repo. This documentation may contain a lot of work-in-progress/outdated/missing
  parts.

## Installation

We highly recommend installing a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`batchtensor` can be installed from pip using the following command:

```shell
uv pip install batchtensor
```

To make the package as slim as possible, only the minimal packages required to use `batchtensor` are
installed.
To include all the dependencies, you can use the following command:

```shell
uv pip install batchtensor[all]
```

Please check the [get started page](https://durandtibo.github.io/batchtensor/get_started) to see how
to install only some specific dependencies or other alternatives to install the library.
The following is the corresponding `batchtensor` versions and tested dependencies.

| `batchtensor` | `coola`          | `numpy`<sup>*</sup> | `torch`       | `python`       |
| ------------- | ---------------- | ------------------- | ------------- | -------------- |
| `main`        | `>=2.0.0a4,<3.0` | `>=1.24,<3.0`       | `>=2.4,<3.0`  | `>=3.11`       |
| `0.1.1`       | `>=0.9.1,<1.0`   | `>=1.24,<2.0`       | `>=2.4,<3.0`  | `>=3.10,<3.15` |
| `0.1.0`       | `>=0.8.6,<1.0`   | `>=1.21,<2.0`       | `>=2.4,<3.0`  | `>=3.9,<3.14`  |
| `0.0.5`       | `>=0.8.6,<1.0`   | `>=1.21,<2.0`       | `>=1.11,<3.0` | `>=3.9,<3.14`  |
| `0.0.4`       | `>=0.1,<1.0`     | `>=1.21,<2.0`       | `>=1.11,<3.0` | `>=3.9,<3.13`  |
| `0.0.3`       | `>=0.1,<1.0`     | `>=1.21,<2.0`       | `>=1.11,<3.0` | `>=3.9,<3.13`  |
| `0.0.2`       | `>=0.1,<1.0`     | `>=1.21,<2.0`       | `>=1.11,<3.0` | `>=3.9,<3.13`  |
| `0.0.1`       | `>=0.1,<0.4`     | `>=1.21,<2.0`       | `>=1.11,<3.0` | `>=3.9,<3.13`  |

<sup>*</sup> indicates an optional dependency

## Contributing

Please check the instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## Suggestions and Communication

Everyone is welcome to contribute to the community.
If you have any questions or suggestions, you can
submit [Github Issues](https://github.com/durandtibo/batchtensor/issues).
We will reply to you as soon as possible. Thank you very much.

## API stability

:warning: While `batchtensor` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `batchtensor` to a new version will possibly break any code
that was using the old version of `batchtensor`.

## License

`batchtensor` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](LICENSE) file.
