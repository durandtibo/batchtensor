r"""Contain code to iterate over the data to find the tensors with a
Depth-First Search (DFS) strategy."""

from __future__ import annotations

__all__ = [
    "BaseTensorIterator",
    "DefaultTensorIterator",
    "IterableTensorIterator",
    "MappingTensorIterator",
    "TensorIterator",
    "dfs_tensor",
    "register_iterators",
    "register_default_iterators",
]

import logging
from collections import deque
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import torch
from coola.utils import str_indent, str_mapping

logger = logging.getLogger(__name__)

T = TypeVar("T")


def dfs_tensor(data: Any) -> Generator[torch.Tensor, None, None]:
    r"""Implement a Depth-First Search (DFS) iterator over the
    ``torch.Tensor``s.

    This function assumes the underlying data has a tree-like
    structure.

    Args:
        data: The data to iterate on.

    Yields:
        The next ``torch.Tensor`` in the data.

    Example usage:

    ```pycon

    >>> import torch
    >>> from batchtensor.utils.dfs import dfs_tensor
    >>> list(dfs_tensor(["abc", torch.ones(2, 3), 42, torch.arange(5)]))
    [tensor([[1., 1., 1.], [1., 1., 1.]]), tensor([0, 1, 2, 3, 4])]

    ```
    """
    state = IteratorState(iterator=TensorIterator())
    yield from state.iterator.iterate(data, state)


@dataclass
class IteratorState:
    r"""Store the current state."""

    iterator: BaseTensorIterator


class BaseTensorIterator(Generic[T]):
    r"""Define the base class to iterate over the data to find the
    tensors with a Depth-First Search (DFS) strategy."""

    def iterate(self, data: T, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        r"""Iterate over the data and add the items to the queue.

        Args:
            data: The data to iterate on.
            state: The current state, which include the
                queue.

        Yields:
            The next ``torch.Tensor`` in the data.
        """


class DefaultTensorIterator(BaseTensorIterator[Any]):
    r"""Implement the default tensor iterator."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(
        self,
        data: Any,
        state: IteratorState,  # noqa: ARG002
    ) -> Generator[torch.Tensor, None, None]:
        if torch.is_tensor(data):
            yield data


class IterableTensorIterator(BaseTensorIterator[Iterable]):
    r"""Implement the tensor iterator for iterable objects."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Iterable, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        for item in data:
            yield from state.iterator.iterate(item, state)


class MappingTensorIterator(BaseTensorIterator[Mapping]):
    r"""Implement the tensor iterator for mapping objects."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Mapping, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        for item in data.values():
            yield from state.iterator.iterate(item, state)


class TensorIterator(BaseTensorIterator[Any]):
    """Implement a tensor iterator."""

    registry: ClassVar[dict[type, BaseTensorIterator]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_iterator(
        cls, data_type: type, iterator: BaseTensorIterator, exist_ok: bool = False
    ) -> None:
        r"""Add an iterator for a given data type.

        Args:
            data_type: The data type for this test.
            iterator: The iterator object.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be set
                to ``True`` to overwrite the iterator for a type.

        Raises:
            RuntimeError: if an iterator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from batchtensor.utils.dfs import TensorIterator, IterableTensorIterator
        >>> TensorIterator.add_iterator(list, IterableTensorIterator(), exist_ok=True)

        ```
        """
        if data_type in cls.registry and not exist_ok:
            msg = (
                f"An iterator ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "iterator for this type"
            )
            raise RuntimeError(msg)
        cls.registry[data_type] = iterator

    def iterate(self, data: Iterable, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        yield from self.find_iterator(type(data)).iterate(data, state)

    @classmethod
    def has_iterator(cls, data_type: type) -> bool:
        r"""Indicate if an iterator is registered for the given data
        type.

        Args:
            data_type: The data type to check.

        Returns:
            ``True`` if an iterator is registered, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from batchtensor.utils.dfs import TensorIterator
        >>> TensorIterator.has_iterator(list)
        True
        >>> TensorIterator.has_iterator(int)
        False

        ```
        """
        return data_type in cls.registry

    @classmethod
    def find_iterator(cls, data_type: Any) -> BaseTensorIterator:
        r"""Find the iterator associated to an object.

        Args:
            data_type: The data type to get.

        Returns:
            The iterator associated to the data type.

        Example usage:

        ```pycon
        >>> from batchtensor.utils.dfs import TensorIterator
        >>> TensorIterator.find_iterator(list)
        IterableTensorIterator()

        ```
        """
        for object_type in data_type.__mro__:
            iterator = cls.registry.get(object_type, None)
            if iterator is not None:
                return iterator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)


def register_iterators(mapping: Mapping[type, BaseTensorIterator]) -> None:
    r"""Register some iterators.

    Args:
        mapping: The iterators to register.
    """
    for typ, op in mapping.items():
        if not TensorIterator.has_iterator(typ):  # pragma: no cover
            TensorIterator.add_iterator(typ, op)


def register_default_iterators() -> None:
    r"""Register some default iterators."""
    default = DefaultTensorIterator()
    iterable = IterableTensorIterator()
    mapping = MappingTensorIterator()
    register_iterators(
        {
            Iterable: iterable,
            Mapping: mapping,
            deque: iterable,
            dict: mapping,
            list: iterable,
            object: default,
            set: iterable,
            str: default,
            tuple: iterable,
        }
    )


register_default_iterators()
