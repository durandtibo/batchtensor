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

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. First, install `uv`
if you haven't already:

```shell
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

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
make setup-venv
source .venv/bin/activate  # On Unix/macOS
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
