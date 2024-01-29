from __future__ import annotations

__all__ = ['dfs']

from abc import ABC
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from coola.utils import str_indent, str_mapping


def dfs(data: Any) -> Iterable[torch.Tensor]:
    pass


class BaseTensorIterator(ABC):

    def __next__(self) -> torch.Tensor:
        pass

class DefaultTensorIterator(BaseTensorIterator):

    def __init__(self, data: Any) -> None:
        self._data = data

    def __next__(self) -> torch.Tensor:
        if torch.is_tensor(self._data):
            yield self._data

class MappingIterator(BaseTensorIterator):
    
    def __init__(self, data: Mapping) -> None:
        self._data = data

    def __next__(self) -> torch.Tensor:
        for value in self._data.values():
            yield value
            
            

class TensorIterator(BaseTensorIterator[Any]):
    """Implement the default equality tester."""

    registry: dict[type, BaseTensorIterator] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_iterator(cls, data_type: type, iterator: BaseTensorIterator, exist_ok: bool = False) -> None:
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
        >>> from batchtensor.iterator.depth import TensorIterator, SequenceTensorIterator
        >>> TensorIterator.add_iterator(list, SequenceTensorIterator(), exist_ok=True)

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

    def apply(self, data: Any, func: Callable, state: ApplyState) -> Any:
        return self.find_iterator(type(data)).apply(data, func, state)

    @classmethod
    def has_iterator(cls, data_type: type) -> bool:
        r"""Indicate if an iterator is registered for the given data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            ``True`` if an iterator is registered, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from batchtensor.iterator.depth import TensorIterator
        >>> TensorIterator.has_iterator(list)
        True
        >>> TensorIterator.has_iterator(str)
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
        >>> from batchtensor.iterator.depth import TensorIterator
        >>> TensorIterator.find_iterator(list)
        SequenceTensorIterator()

        ```
        """
        for object_type in data_type.__mro__:
            iterator = cls.registry.get(object_type, None)
            if iterator is not None:
                return iterator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)