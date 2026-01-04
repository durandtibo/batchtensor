# Get Started

This guide will help you install `batchtensor` and verify your installation.

## Prerequisites

`batchtensor` requires:

- Python 3.10 or later
- PyTorch 2.4 or later
- A compatible operating system (Linux, macOS, or Windows)

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `uv pip` (recommended)

The following command installs the latest version of the library:

```shell
uv pip install batchtensor
```

To make the package as slim as possible, only the packages required to use `batchtensor` are
installed.
It is possible to install all the optional dependencies by running the following command:

```shell
uv pip install 'batchtensor[all]'
```

## Installing from source

To install `batchtensor` from source, you can follow the steps below.

### Prerequisites

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
Please refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

You can verify the installation by running:

```shell
uv --version
```

### Clone the repository

```shell
git clone git@github.com:durandtibo/batchtensor.git
cd batchtensor
```

### Create a virtual environment

It is recommended to create a Python 3.10+ virtual environment.
You can create a virtual environment with `uv`:

```shell
inv create-venv
```

Alternatively, you can use the Makefile shortcut which also installs all dependencies:

```shell
make setup-venv
source .venv/bin/activate
```

### Install dependencies

Install all dependencies using `uv`:

```shell
inv install
```

To install with documentation dependencies:

```shell
inv install --docs-deps
```

### Verify the installation

Run the test suite to verify everything is working:

```shell
inv unit-test --cov
```

## Next Steps

After installation, explore the documentation:

- **[Tensor Operations Guide](uguide/tensor.md)**: Learn about single tensor operations
- **[Nested Operations Guide](uguide/nested.md)**: Learn about nested structure operations
- **[Utils Guide](uguide/utils.md)**: Learn about utility functions
- **[API Reference](refs/nested.md)**: Browse the complete API

## Quick Example

Here's a simple example to verify your installation:

```pycon
>>> import torch
>>> from batchtensor.nested import slice_along_batch
>>> batch = {
...     "features": torch.tensor([[1, 2], [3, 4], [5, 6]]),
...     "labels": torch.tensor([0, 1, 2]),
... }
>>> # Take the first 2 samples
>>> slice_along_batch(batch, stop=2)
{'features': tensor([[1, 2], [3, 4]]), 'labels': tensor([0, 1])}

```

If this runs without errors, your installation is successful!

## Troubleshooting

### Import Errors

If you encounter import errors, ensure that:

1. You're using Python 3.10 or later
2. PyTorch is properly installed
3. Your virtual environment is activated (if using one)

### PyTorch Installation

If PyTorch is not installed, install it following the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Getting Help

If you encounter any issues:

- Check the [GitHub Issues](https://github.com/durandtibo/batchtensor/issues) for known problems
- Create a new issue if your problem is not already reported
- Review the [documentation](https://durandtibo.github.io/batchtensor/) for detailed usage
  information
