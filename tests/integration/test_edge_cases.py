r"""Pin the documented result of the edge cases (P16, ``design_principles.md``).

A check that needs the values of a tensor would need a host-device sync, so
the functions do not make it. The docstrings state the result instead, and
this table fails when it changes.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor import nested, tensor as bt

FLOAT_LOWEST = torch.finfo(torch.float32).min
FLOAT_HIGHEST = torch.finfo(torch.float32).max
INT_LOWEST = torch.iinfo(torch.int64).min
INT_HIGHEST = torch.iinfo(torch.int64).max

FLOATS = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
INTS = torch.tensor([[1, 2], [3, 4]])
# The second row has no valid step.
MASK = torch.tensor([[True, True], [False, False]])

EDGE_CASES = [
    pytest.param(
        lambda: bt.masked_mean_along_seq(FLOATS, MASK),
        torch.tensor([1.5, torch.nan]),
        id="masked-mean-fully-masked-row",
    ),
    pytest.param(
        lambda: bt.masked_sum_along_seq(FLOATS, MASK),
        torch.tensor([3.0, 0.0]),
        id="masked-sum-fully-masked-row",
    ),
    pytest.param(
        lambda: bt.masked_amax_along_seq(FLOATS, MASK),
        torch.tensor([2.0, FLOAT_LOWEST]),
        id="masked-amax-fully-masked-row-float",
    ),
    pytest.param(
        lambda: bt.masked_amin_along_seq(FLOATS, MASK),
        torch.tensor([1.0, FLOAT_HIGHEST]),
        id="masked-amin-fully-masked-row-float",
    ),
    pytest.param(
        lambda: bt.masked_amax_along_seq(INTS, MASK),
        torch.tensor([2, INT_LOWEST]),
        id="masked-amax-fully-masked-row-int",
    ),
    pytest.param(
        lambda: bt.masked_amin_along_seq(INTS, MASK),
        torch.tensor([1, INT_HIGHEST]),
        id="masked-amin-fully-masked-row-int",
    ),
    pytest.param(
        lambda: nested.masked_amax_along_seq({"a": INTS}, MASK),
        {"a": torch.tensor([2, INT_LOWEST])},
        id="masked-amax-fully-masked-row-nested",
    ),
    pytest.param(
        lambda: bt.permute_along_batch(INTS, torch.tensor([0, 0])),
        torch.tensor([[1, 2], [1, 2]]),
        id="permute-batch-repeated-index",
    ),
    pytest.param(
        lambda: bt.permute_along_seq(INTS, torch.tensor([1, 1])),
        torch.tensor([[2, 2], [4, 4]]),
        id="permute-seq-repeated-index",
    ),
    pytest.param(
        lambda: nested.permute_along_batch({"a": INTS}, torch.tensor([0, 0])),
        {"a": torch.tensor([[1, 2], [1, 2]])},
        id="permute-batch-repeated-index-nested",
    ),
    pytest.param(
        lambda: bt.lengths_to_mask(torch.tensor([-1, 2])),
        torch.tensor([[False, False], [True, True]]),
        id="lengths-to-mask-negative-length",
    ),
    pytest.param(
        lambda: bt.lengths_to_mask(torch.tensor([3, 2]), max_len=2),
        torch.tensor([[True, True], [True, True]]),
        id="lengths-to-mask-max-len-below-length",
    ),
    pytest.param(
        lambda: bt.pad_along_seq(INTS, 0),
        torch.tensor([[1, 2], [3, 4]]),
        id="pad-seq-zero-size",
    ),
    pytest.param(
        lambda: bt.pad_along_batch(INTS, 0),
        torch.tensor([[1, 2], [3, 4]]),
        id="pad-batch-zero-size",
    ),
    pytest.param(
        lambda: bt.pad_sequences([torch.zeros(0, dtype=torch.long), torch.tensor([1])], value=7),
        torch.tensor([[7], [1]]),
        id="pad-sequences-zero-length-tensor",
    ),
    pytest.param(
        lambda: bt.pad_sequences([torch.zeros(0, dtype=torch.long)], value=7),
        torch.zeros(1, 0, dtype=torch.long),
        id="pad-sequences-only-zero-length",
    ),
    pytest.param(
        lambda: nested.pad_sequences(
            [{"a": torch.zeros(0, dtype=torch.long)}, {"a": torch.tensor([1])}], value=7
        ),
        {"a": torch.tensor([[7], [1]])},
        id="pad-sequences-zero-length-tensor-nested",
    ),
]


@pytest.mark.parametrize(("call", "expected"), EDGE_CASES)
def test_edge_case(call: Any, expected: Any) -> None:
    assert objects_are_equal(call(), expected, equal_nan=True)
