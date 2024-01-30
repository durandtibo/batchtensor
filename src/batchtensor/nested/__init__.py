r"""Contain functions to manipulate nested data."""

from __future__ import annotations

__all__ = [
    "cat_along_batch",
    "cat_along_seq",
    "chunk_along_batch",
    "chunk_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "select_along_batch",
    "select_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
    "slice_along_batch",
    "slice_along_seq",
    "split_along_batch",
    "split_along_seq",
]

from batchtensor.nested.indexing import index_select_along_batch, index_select_along_seq
from batchtensor.nested.joining import cat_along_batch, cat_along_seq
from batchtensor.nested.permutation import (
    permute_along_batch,
    permute_along_seq,
    shuffle_along_batch,
    shuffle_along_seq,
)
from batchtensor.nested.slicing import (
    chunk_along_batch,
    chunk_along_seq,
    select_along_batch,
    select_along_seq,
    slice_along_batch,
    slice_along_seq,
    split_along_batch,
    split_along_seq,
)
