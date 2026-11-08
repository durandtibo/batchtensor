r"""Gates for the properties of ``design_principles.md`` that no other test
enforces: P1 (no ``dim`` argument), P4 (one traversal) and P7 (inputs are not
modified)."""

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
}
JOINING = {"cat", "stack"}
SKIPPED = {"nested.from_numpy", "nested.pin_memory", "tensor.lengths_to_mask"}


def get_operation(name: str) -> str:
    r"""Return the operation of a function (the name without the dimension)."""
    return name.split(".")[1].split("_along_")[0]


def get_inputs(name: str) -> tuple[list[Any], dict[str, Any]]:
    r"""Return the positional and keyword arguments of a function."""
    x = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    operation = get_operation(name)
    args, kwargs = OPERATION_ARGS.get(operation, ((), {}))
    is_nested = name.startswith("nested.")
    data: Any = {"a": x, "b": x + 1} if is_nested else x
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
