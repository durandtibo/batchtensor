from __future__ import annotations

__all__ = ["bfs_tensor"]

import logging
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch
from coola.utils import str_indent, str_mapping

logger = logging.getLogger(__name__)

T = TypeVar("T")


def bfs_tensor(data: Any) -> Generator[torch.Tensor, None, None]:
    pass



@dataclass
class IteratorState:
    r"""Store the current state."""

    iterator: BaseTensorIterator
    queue: list = field(default_factory=list)



class BaseTensorIterator(Generic[T]):

    def iterate(self, data: T, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        pass


class DefaultTensorIterator(BaseTensorIterator[Any]):

    def iterate(self, data: Any, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        if torch.is_tensor(data):
            yield data


class IterableTensorIterator(BaseTensorIterator[Iterable]):

    def iterate(self, data: Iterable, state: IteratorState) -> Generator[torch.Tensor, None, None]:
        for item in data:
            yield from state.iterator.iterate(item, state=state)