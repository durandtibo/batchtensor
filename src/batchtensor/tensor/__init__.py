r"""Contain functions to manipulate tensors."""

from __future__ import annotations

__all__ = [
    "amax_along_batch",
    "amax_along_seq",
    "amin_along_batch",
    "amin_along_seq",
    "argmax_along_batch",
    "argmax_along_seq",
    "argmin_along_batch",
    "argmin_along_seq",
    "argsort_along_batch",
    "argsort_along_seq",
    "cat_along_batch",
    "cat_along_seq",
    "chunk_along_batch",
    "chunk_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "repeat_along_seq",
    "select_along_batch",
    "select_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
    "slice_along_batch",
    "slice_along_seq",
    "split_along_batch",
    "split_along_seq",
    "sum_along_batch",
    "sum_along_seq",
]

from batchtensor.tensor.comparison import argsort_along_batch, argsort_along_seq
from batchtensor.tensor.indexing import index_select_along_batch, index_select_along_seq
from batchtensor.tensor.joining import cat_along_batch, cat_along_seq, repeat_along_seq
from batchtensor.tensor.permutation import (
    permute_along_batch,
    permute_along_seq,
    shuffle_along_batch,
    shuffle_along_seq,
)
from batchtensor.tensor.reduction import (
    amax_along_batch,
    amax_along_seq,
    amin_along_batch,
    amin_along_seq,
    argmax_along_batch,
    argmax_along_seq,
    argmin_along_batch,
    argmin_along_seq,
    sum_along_batch,
    sum_along_seq,
)
from batchtensor.tensor.slicing import (
    chunk_along_batch,
    chunk_along_seq,
    select_along_batch,
    select_along_seq,
    slice_along_batch,
    slice_along_seq,
    split_along_batch,
    split_along_seq,
)
