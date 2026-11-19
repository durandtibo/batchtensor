r"""Gate for the error contract (P8) of ``design_principles.md``.

Each row of the table of P8 has at least one case: a call that is expected
to raise the exception of the contract.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from batchtensor import nested, tensor as bt

BAD_LEAF: Any = "x"
BAD_SIDE: Any = "up"
BAD_FLOAT_VALUE: Any = 1.5


def ones(*shape: int, dtype: torch.dtype = torch.float) -> torch.Tensor:
    return torch.ones(*shape, dtype=dtype)


DIFFERENT_LENGTHS: Any = [[ones(2, 3)], [ones(2, 3), ones(2, 3)]]

# (call, exception, regular expression of the message)
CASES = [
    # A leaf is not a ``torch.Tensor``
    pytest.param(
        lambda: nested.sum_along_batch({"a": BAD_LEAF}),
        TypeError,
        r"Expected a torch.Tensor leaf but received str",
        id="leaf-map",
    ),
    pytest.param(
        lambda: nested.chunk_along_batch({"a": BAD_LEAF}, 2),
        TypeError,
        r"Expected a torch.Tensor leaf but received str",
        id="leaf-split",
    ),
    pytest.param(
        lambda: nested.cat_along_batch([{"a": BAD_LEAF}, {"a": BAD_LEAF}]),
        TypeError,
        r"Expected a torch.Tensor leaf but received str",
        id="leaf-combine",
    ),
    pytest.param(
        lambda: nested.to_numpy({"a": BAD_LEAF}),
        TypeError,
        r"Expected a torch.Tensor leaf but received str",
        id="leaf-to_numpy",
    ),
    # ``cat``/``stack`` receives no item or structures with different layout
    pytest.param(
        lambda: nested.cat_along_batch([]),
        ValueError,
        r"items must contain at least one structure",
        id="combine-empty",
    ),
    pytest.param(
        lambda: nested.stack_along_batch([{"a": ones(2)}, {"b": ones(2)}]),
        ValueError,
        r"The mappings must have the same keys",
        id="combine-different-keys",
    ),
    pytest.param(
        lambda: nested.cat_along_seq(DIFFERENT_LENGTHS),
        ValueError,
        r"The containers must have the same length",
        id="combine-different-lengths",
    ),
    # Leaves disagree on the split, chunk or shuffle dimension
    pytest.param(
        lambda: nested.split_along_batch({"a": ones(4, 3), "b": ones(6, 3)}, 2),
        ValueError,
        r"All the tensors must be split into the same number of pieces",
        id="split-different-sizes",
    ),
    pytest.param(
        lambda: nested.chunk_along_seq({"a": ones(2, 4), "b": ones(2, 1)}, 4),
        ValueError,
        r"All the tensors must be split into the same number of pieces",
        id="chunk-different-sizes",
    ),
    pytest.param(
        lambda: nested.shuffle_along_batch({"a": ones(4, 3), "b": ones(5, 3)}),
        ValueError,
        r"All the tensors must have the same size along dimension 0",
        id="shuffle-batch-different-sizes",
    ),
    pytest.param(
        lambda: nested.shuffle_along_seq({"a": ones(2, 3), "b": ones(2, 4)}),
        ValueError,
        r"All the tensors must have the same size along dimension 1",
        id="shuffle-seq-different-sizes",
    ),
    # ``pad_size < 0``, a ``value`` that does not fit the dtype, a bad ``side``
    pytest.param(
        lambda: bt.pad_along_batch(ones(4, 3), -1),
        ValueError,
        r"pad_size must be a non-negative integer but received -1",
        id="pad-negative-size",
    ),
    pytest.param(
        lambda: nested.pad_along_seq({"a": ones(4, 3)}, -1),
        ValueError,
        r"pad_size must be a non-negative integer but received -1",
        id="pad-negative-size-nested",
    ),
    pytest.param(
        lambda: bt.pad_along_batch(ones(4, 3, dtype=torch.long), 1, value=BAD_FLOAT_VALUE),
        ValueError,
        r"value \(1.5\) cannot be represented with dtype torch.int64",
        id="pad-value-does-not-fit",
    ),
    pytest.param(
        lambda: bt.pad_along_seq(ones(4, 3), 1, side=BAD_SIDE),
        ValueError,
        r"side must be 'left' or 'right' but received 'up'",
        id="pad-bad-side",
    ),
    pytest.param(
        lambda: bt.pad_sequences([ones(2, dtype=torch.long), ones(1)]),
        ValueError,
        r"tensors must have the same dtype and device",
        id="pad-sequences-mixed-dtype",
    ),
    pytest.param(
        lambda: nested.pad_sequences([{"a": ones(2, dtype=torch.long)}, {"a": ones(1)}]),
        ValueError,
        r"tensors must have the same dtype and device",
        id="pad-sequences-mixed-dtype-nested",
    ),
    # A ``permutation``/``index`` with the wrong ndim or length
    pytest.param(
        lambda: bt.permute_along_batch(ones(4, 3), torch.zeros(2, 2, dtype=torch.long)),
        ValueError,
        r"permutation shape .* is not compatible with tensor shape",
        id="permutation-wrong-ndim",
    ),
    pytest.param(
        lambda: nested.permute_along_seq({"a": ones(4, 3)}, torch.arange(2)),
        ValueError,
        r"permutation shape .* is not compatible with tensor shape",
        id="permutation-wrong-length",
    ),
    pytest.param(
        lambda: bt.index_select_along_batch(ones(4, 3), torch.zeros(2, 2, dtype=torch.long)),
        ValueError,
        r"index must be a 1-D tensor but received a tensor with 2 dimensions",
        id="index-wrong-ndim",
    ),
    # ``lengths`` is not a 1-D integer tensor
    pytest.param(
        lambda: bt.lengths_to_mask(torch.ones(2, 2, dtype=torch.long)),
        ValueError,
        r"lengths must be a 1-D tensor but received a tensor with 2 dimensions",
        id="lengths-wrong-ndim",
    ),
    pytest.param(
        lambda: bt.lengths_to_mask(torch.ones(2)),
        ValueError,
        r"lengths must be an integer tensor but received dtype torch.float32",
        id="lengths-not-integer",
    ),
    # ``repeats < 0``, ``chunks <= 0``, a bad dtype: the torch error, unchanged
    pytest.param(
        lambda: bt.repeat_along_batch(ones(4, 3), -1),
        RuntimeError,
        r"negative dimension",
        id="torch-negative-repeats",
    ),
    pytest.param(
        lambda: bt.chunk_along_batch(ones(4, 3), 0),
        RuntimeError,
        r"chunk expects `chunks` to be greater than 0",
        id="torch-zero-chunks",
    ),
    pytest.param(
        lambda: nested.mean_along_batch({"a": torch.arange(3)}),
        NotImplementedError,
        r"could not infer output dtype",
        id="torch-integer-mean",
    ),
]


@pytest.mark.parametrize(("call", "exception", "match"), CASES)
def test_error_contract(call: Any, exception: type[Exception], match: str) -> None:
    with pytest.raises(exception, match=match):
        call()
