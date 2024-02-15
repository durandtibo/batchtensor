from __future__ import annotations

__all__ = ["IteratorState"]

import logging
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from coola.utils import str_indent, str_mapping

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class IteratorState:
    r"""Store the current state."""

    iterator: BaseTensorIterator
    depth: int = 0

    def increment_depth(self, increment: int = 1) -> IteratorState:
        return IteratorState(iterator=self.iterator, depth=self.depth + increment)


class BaseTensorIterator(Generic[T]):

    def iterate(self, data: T, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        pass


class DefaultTensorIterator(BaseTensorIterator[Any]):

    def iterate(self, data: Any, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        if torch.is_tensor(data):
            yield data


class DBSTensorIterator(BaseTensorIterator[Iterable]):

    def iterate(self, data: Iterable, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        for item in data:
            yield from state.iterator.iterate(item, state=state.increment_depth())


class TensorIterator(BaseTensorIterator[Any]):
    """Implement the default equality tester."""

    registry: dict[type, BaseTensorIterator] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_iterator(
            cls, data_type: type, iterator: BaseTensorIterator, exist_ok: bool = False
    ) -> None:
        r"""Add an iterator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            iterator: Specifies the iterator object.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be set
                to ``True`` to overwrite the iterator for a type.

        Raises:
            RuntimeError: if a iterator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from batchtensor.recursive import Applier, SequenceApplier
        >>> Applier.add_iterator(list, SequenceApplier(), exist_ok=True)

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

    def iterate(self, data: Any, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        yield from self.find_iterator(type(data)).iterate(data, state)

    @classmethod
    def has_iterator(cls, data_type: type) -> bool:
        r"""Indicate if an iterator is registered for the given data
        type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            ``True`` if an iterator is registered, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from batchtensor.recursive import Applier
        >>> Applier.has_iterator(list)
        True
        >>> Applier.has_iterator(str)
        False

        ```
        """
        return data_type in cls.registry

    @classmethod
    def find_iterator(cls, data_type: Any) -> BaseTensorIterator:
        r"""Find the iterator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            The iterator associated to the data type.

        Example usage:

        ```pycon
        >>> from batchtensor.recursive import Applier
        >>> Applier.find_iterator(list)
        SequenceApplier()

        ```
        """
        for object_type in data_type.__mro__:
            iterator = cls.registry.get(object_type, None)
            if iterator is not None:
                return iterator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)


def register_iterators(mapping: Mapping[type, BaseTensorIterator]) -> None:
    for typ, op in mapping.items():
        if not TensorIterator.has_iterator(typ):  # pragma: no cover
            TensorIterator.add_iterator(typ, op)


if __name__ == "__main__":
    register_iterators(
        {
            object: DefaultTensorIterator(),
            torch.Tensor: DefaultTensorIterator(),
            list: DBSTensorIterator(),
            tuple: DBSTensorIterator(),
            Iterable: DBSTensorIterator(),
        }
    )
    state = IteratorState(iterator=TensorIterator())
    print(state.iterator)

    data = {
        "int": 1,
        "list": [1, torch.ones(2, 3), "abc"],
        "tensor": torch.arange(3),
        "dict": {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([7, 8, 9])},
    }

    print(list(state.iterator.iterate([1, torch.ones(2, 3), "abc"], state=state)))

    # def iterator1(n: int = 10):
    #     yield from range(n)
    #
    # def iterator2(n: int = 10):
    #     for i in range(n):
    #         yield from iterator1(i)
    #
    # print(list(iterator2()))
    #
    # gen = iterator2()
    # for _ in range(5):
    #     print(next(gen))
