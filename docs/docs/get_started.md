# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `pip` (recommended)

The following command installs the latest version of the library:

```shell
pip install batchtensor
```

To make the package as slim as possible, only the packages required to use `batchtensor` are
installed.
It is possible to install all the optional dependencies by running the following command:

```shell
pip install 'batchtensor[all]'
```

## Installing from source

To install `batchtensor` from source, you can follow the steps below.

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. First, install `uv`:

```shell
pip install uv
```

### Clone the repository

```shell
git clone git@github.com:durandtibo/batchtensor.git
cd batchtensor
```

### Create a virtual environment

It is recommended to create a Python 3.10+ virtual environment. You can use `conda`:

```shell
make conda
conda activate batchtensor
```

Or create a virtual environment with `uv`:

```shell
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Install dependencies

Install all dependencies using `uv`:

```shell
uv sync --frozen
```

Or use the Makefile:

```shell
make install
```

To install with documentation dependencies:

```shell
make install-all
```

### Verify the installation

Run the test suite to verify everything is working:

```shell
make unit-test-cov
```

Or use `uv` directly:

```shell
uv run pytest tests/
```
