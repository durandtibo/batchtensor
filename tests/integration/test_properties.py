r"""Gates for the properties of ``design_principles.md`` that no other
test enforces: P1 (no ``dim`` argument), P4 (one traversal) and P7
(inputs are not modified, outputs do not alias the inputs), P2 (the
closed list of functions that are not one torch call) and P10 (no cast)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor import nested, tensor

TENSOR_FUNCTIONS = [(f"tensor.{name}", getattr(tensor, name)) for name in tensor.__all__]
NESTED_FUNCTIONS = [
    (f"nested.{name}", getattr(nested, name)) for name in nested.__all__ if name != "NestedTensor"
]
ALL_FUNCTIONS = TENSOR_FUNCTIONS + NESTED_FUNCTIONS

NESTED_DIR = Path(nested.__file__).parent

##########################################
#     Tests for P1 (no dim argument)     #
##########################################


@pytest.mark.parametrize("fn", [f for _, f in ALL_FUNCTIONS], ids=[n for n, _ in ALL_FUNCTIONS])
def test_no_dim_parameter(fn: Any) -> None:
    assert "dim" not in inspect.signature(fn).parameters


########################################
#     Tests for P4 (one traversal)     #
########################################

# The ``nested`` functions that are allowed to walk a container themselves.
# ``from_numpy`` has array leaves, so it cannot use ``apply_to_tensors``.
TRAVERSAL_ALLOWED = {"conversion.py": ["from_numpy"]}
CONTAINER_NAMES = {"dict", "list", "tuple", "set", "Mapping", "Sequence", "Iterable"}
TRAVERSAL_CALLS = {"recursive_apply", "dfs_iterate", "rebuild_mapping", "rebuild_sequence"}


def get_called_name(node: ast.Call) -> str:
    r"""Return the name of the called function, or an empty string."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def has_container_check(node: ast.Call) -> bool:
    r"""Indicate if a call is ``isinstance(x, <container types>)``."""
    if get_called_name(node) != "isinstance" or len(node.args) < 2:
        return False
    names = {n.id for n in ast.walk(node.args[1]) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(node.args[1]) if isinstance(n, ast.Attribute)}
    return bool(names & CONTAINER_NAMES)


def find_traversals(path: Path) -> list[str]:
    r"""Return the functions of a module that walk a container."""
    tree = ast.parse(path.read_text())
    found = []
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)]
        if any(get_called_name(c) in TRAVERSAL_CALLS or has_container_check(c) for c in calls):
            found.append(func.name)
    return sorted(found)


NESTED_MODULES = sorted(NESTED_DIR.glob("*.py"))


@pytest.mark.parametrize("path", NESTED_MODULES, ids=[p.name for p in NESTED_MODULES])
def test_nested_module_does_not_walk_containers(path: Path) -> None:
    assert find_traversals(path) == TRAVERSAL_ALLOWED.get(path.name, [])


def test_find_traversals_detects_recursive_apply(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("def f(data):\n    return recursive_apply(data, g)\n")
    assert find_traversals(path) == ["f"]


def test_find_traversals_detects_isinstance_container(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("def f(data):\n    return isinstance(data, (dict, list))\n")
    assert find_traversals(path) == ["f"]


def test_find_traversals_ignores_isinstance_tensor(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("def f(data):\n    return isinstance(data, torch.Tensor)\n")
    assert find_traversals(path) == []


######################################
#     Tests for P7 (pure inputs)     #
######################################

# The arguments (after the data) and the keyword arguments of each operation.
# The data have the shape (3, 3, 2), so that the same permutation is valid
# along the batch and the sequence dimensions.
OPERATION_ARGS: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
    "chunk": ((2,), {}),
    "index_select": ((torch.tensor([0, 2]),), {}),
    "pad": ((1,), {}),
    "permute": ((torch.tensor([2, 0, 1]),), {}),
    "repeat": ((2,), {}),
    "roll": ((1,), {}),
    "select": ((0,), {}),
    "slice": ((0, 2), {}),
    "split": ((2,), {}),
    "topk": ((2,), {}),
    "to": ((), {"dtype": torch.float64}),
    **{
        f"masked_{name}": ((torch.ones(3, 3, dtype=torch.bool),), {})
        for name in ("sum", "mean", "amax", "amin")
    },
}
JOINING = {"cat", "stack"}
SKIPPED = {"nested.from_numpy", "nested.pin_memory", "tensor.lengths_to_mask"}


def get_operation(name: str) -> str:
    r"""Return the operation of a function (the name without the
    dimension)."""
    return name.split(".")[1].split("_along_")[0]


def get_inputs(name: str) -> tuple[list[Any], dict[str, Any]]:
    r"""Return the positional and keyword arguments of a function."""
    x = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    operation = get_operation(name)
    args, kwargs = OPERATION_ARGS.get(operation, ((), {}))
    is_nested = name.startswith("nested.")
    data: Any = {"a": x, "b": x + 1} if is_nested else x
    if operation == "pad_sequences" and is_nested:
        return [[data, {"a": x[:2], "b": x[:2] + 1}], *args], kwargs
    if operation == "pad_sequences":
        return [[x, x[:2]], *args], kwargs
    if operation in JOINING:
        return [[data, data.copy() if is_nested else data.clone()], *args], kwargs
    return [data, *args], kwargs


def snapshot(value: Any) -> Any:
    r"""Return a deep copy of the tensors of a nested value."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: snapshot(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(val) for val in value]
    return value


PURE_CASES = [(name, fn) for name, fn in ALL_FUNCTIONS if name not in SKIPPED]


@pytest.mark.parametrize(("name", "fn"), PURE_CASES, ids=[name for name, _ in PURE_CASES])
def test_inputs_are_not_modified(name: str, fn: Any) -> None:
    args, kwargs = get_inputs(name)
    before = snapshot(args)
    fn(*args, **kwargs)
    assert objects_are_equal(snapshot(args), before)


#####################################################
#     Tests for P15 (the layout is stated once)     #
#####################################################


@pytest.mark.parametrize("fn", [f for _, f in ALL_FUNCTIONS], ids=[n for n, _ in ALL_FUNCTIONS])
def test_docstring_does_not_repeat_layout(fn: Any) -> None:
    assert "This function assumes the" not in (inspect.getdoc(fn) or "")


############################################
#     Tests for P7 (no aliasing)           #
############################################

# Functions whose output may share memory with the input: the views of torch
# (``slice``, ``select``, ``chunk``, ``split``, ``detach``, ``to_numpy`` on
# CPU), ``from_numpy`` and the functions that return the input when nothing
# changes (``to``, ``contiguous``, ``pin_memory``).
MAY_ALIAS = {
    "slice",
    "select",
    "chunk",
    "split",
    "detach",
    "to",
    "contiguous",
    "pin_memory",
    "to_numpy",
    "from_numpy",
}


def get_tensors(value: Any) -> list[torch.Tensor]:
    r"""Return the tensors of a nested value."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for val in value.values() for t in get_tensors(val)]
    if isinstance(value, (list, tuple)):
        return [t for val in value for t in get_tensors(val)]
    return []


def shares_memory(inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> bool:
    r"""Indicate if an output tensor shares a storage with an input tensor."""
    storages = {t.untyped_storage().data_ptr() for t in inputs}
    return any(t.untyped_storage().data_ptr() in storages for t in outputs)


ALIAS_CASES = [(name, fn) for name, fn in PURE_CASES if get_operation(name) not in MAY_ALIAS]


@pytest.mark.parametrize(("name", "fn"), ALIAS_CASES, ids=[name for name, _ in ALIAS_CASES])
def test_output_does_not_alias_input(name: str, fn: Any) -> None:
    args, kwargs = get_inputs(name)
    out = fn(*args, **kwargs)
    assert not shares_memory(get_tensors(args), get_tensors(out))


#######################################################
#     Tests for P2 (the closed list of exceptions)    #
#######################################################

TENSOR_DIR = Path(tensor.__file__).parent

# The ``tensor`` functions that are not one torch call
# (``design_principles.md``).
NOT_THIN = {
    "masked_sum_along_seq",
    "masked_mean_along_seq",
    "masked_amax_along_seq",
    "masked_amin_along_seq",
    "lengths_to_mask",
    "pad_sequences",
    "pad_along_batch",
    "pad_along_seq",
    "repeat_along_batch",
    "repeat_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
}


def is_one_torch_call(node: ast.FunctionDef) -> bool:
    r"""Indicate if a function is a docstring and one ``return`` of a call
    on ``torch`` or on the tensor, or of an index of the tensor."""
    body = [
        stmt
        for stmt in node.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if isinstance(value, ast.Subscript):
        value = value.value
    elif isinstance(value, ast.Call):
        value = value.func
    else:
        return False
    while isinstance(value, (ast.Attribute, ast.Call, ast.Subscript)):
        value = value.func if isinstance(value, ast.Call) else value.value
    return isinstance(value, ast.Name) and value.id in {"torch", "tensor"}


def get_tensor_functions() -> dict[str, ast.FunctionDef]:
    r"""Return the AST of the public functions of the ``tensor`` package."""
    functions = {}
    for path in sorted(TENSOR_DIR.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in tensor.__all__:
                functions[node.name] = node
    return functions


TENSOR_ASTS = get_tensor_functions()


@pytest.mark.parametrize("name", sorted(set(TENSOR_ASTS) - NOT_THIN))
def test_thin_wrappers(name: str) -> None:
    assert is_one_torch_call(TENSOR_ASTS[name])


@pytest.mark.parametrize("name", sorted(NOT_THIN))
def test_not_thin_list_has_no_stale_entry(name: str) -> None:
    assert name in TENSOR_ASTS


###################################
#     Tests for P10 (no cast)     #
###################################

# The functions whose output is not of the dtype of the input.
INDEX_OPERATIONS = {"argmax", "argmin", "argsort"}
BOOL_OPERATIONS = {"any", "all"}
# The functions that return the values and the indices.
PAIR_OPERATIONS = {"max", "min", "median", "sort", "topk"}
NO_CAST_SKIPPED = SKIPPED | {"tensor.to", "nested.to", "nested.to_numpy"}


def to_float64(value: Any) -> Any:
    r"""Cast the floating point tensors of a nested value to float64."""
    if isinstance(value, torch.Tensor):
        return value.double() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: to_float64(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_float64(val) for val in value]
    return value


def get_expected_dtypes(operation: str) -> set[torch.dtype]:
    r"""Return the dtypes allowed in the output of an operation."""
    if operation in INDEX_OPERATIONS:
        return {torch.int64}
    if operation in BOOL_OPERATIONS:
        return {torch.bool}
    if operation in PAIR_OPERATIONS:
        return {torch.float64, torch.int64}
    if operation == "lengths_to_mask":
        return {torch.bool}
    return {torch.float64}


NO_CAST_CASES = [(name, fn) for name, fn in ALL_FUNCTIONS if name not in NO_CAST_SKIPPED]


@pytest.mark.parametrize(("name", "fn"), NO_CAST_CASES, ids=[name for name, _ in NO_CAST_CASES])
def test_output_dtype_is_the_one_of_torch(name: str, fn: Any) -> None:
    args, kwargs = get_inputs(name)
    out = fn(*to_float64(args), **kwargs)
    dtypes = {t.dtype for t in get_tensors(out)}
    assert dtypes <= get_expected_dtypes(get_operation(name))
