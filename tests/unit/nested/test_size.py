from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from batchtensor.nested import batch_size, seq_len

################################
#     Tests for batch_size     #
################################


def test_batch_size_tensor() -> None:
    assert batch_size(torch.ones(2, 3)) == 2


def test_batch_size_1d_tensor() -> None:
    assert batch_size(torch.ones(4)) == 4


def test_batch_size_dict() -> None:
    assert batch_size({"a": torch.ones(2, 3), "b": torch.ones(2)}) == 2


def test_batch_size_list() -> None:
    assert batch_size([torch.ones(5, 3), torch.ones(5)]) == 5


def test_batch_size_tuple() -> None:
    assert batch_size((torch.ones(3, 3), torch.ones(3))) == 3


def test_batch_size_mixed_nest() -> None:
    assert (
        batch_size({"a": [torch.ones(2, 3), (torch.ones(2),)], "b": {"c": torch.ones(2, 1)}}) == 2
    )


def test_batch_size_returns_int() -> None:
    assert type(batch_size(torch.ones(2, 3))) is int


def test_batch_size_empty_batch() -> None:
    assert batch_size(torch.ones(0, 3)) == 0


def test_batch_size_no_tensor() -> None:
    with pytest.raises(ValueError, match=r"does not contain any tensor"):
        batch_size({})


def test_batch_size_disagreeing_leaves() -> None:
    with pytest.raises(ValueError, match=r"must have the same size along dimension"):
        batch_size({"a": torch.ones(2, 3), "b": torch.ones(3, 3)})


def test_batch_size_scalar_leaf() -> None:
    with pytest.raises(ValueError, match=r"at least 1 dimension\(s\)"):
        batch_size(torch.tensor(1.0))


def test_batch_size_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        batch_size(cast("Any", {"a": "abc"}))


def test_batch_size_does_not_modify_input() -> None:
    data = {"a": torch.ones(2, 3)}
    batch_size(data)
    assert data["a"].equal(torch.ones(2, 3))


#############################
#     Tests for seq_len     #
#############################


def test_seq_len_tensor() -> None:
    assert seq_len(torch.ones(2, 3)) == 3


def test_seq_len_dict() -> None:
    assert seq_len({"a": torch.ones(2, 3), "b": torch.ones(4, 3, 5)}) == 3


def test_seq_len_list() -> None:
    assert seq_len([torch.ones(2, 6), torch.ones(2, 6, 1)]) == 6


def test_seq_len_tuple() -> None:
    assert seq_len((torch.ones(2, 4), torch.ones(2, 4))) == 4


def test_seq_len_mixed_nest() -> None:
    assert (
        seq_len({"a": [torch.ones(2, 3), (torch.ones(1, 3, 2),)], "b": {"c": torch.ones(2, 3)}})
        == 3
    )


def test_seq_len_returns_int() -> None:
    assert type(seq_len(torch.ones(2, 3))) is int


def test_seq_len_empty_sequence() -> None:
    assert seq_len(torch.ones(2, 0)) == 0


def test_seq_len_no_tensor() -> None:
    with pytest.raises(ValueError, match=r"does not contain any tensor"):
        seq_len([])


def test_seq_len_disagreeing_leaves() -> None:
    with pytest.raises(ValueError, match=r"must have the same size along dimension"):
        seq_len({"a": torch.ones(2, 3), "b": torch.ones(2, 4)})


def test_seq_len_1d_leaf() -> None:
    with pytest.raises(ValueError, match=r"at least 2 dimension\(s\)"):
        seq_len({"a": torch.ones(2, 3), "b": torch.ones(2)})


def test_seq_len_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received int"):
        seq_len(cast("Any", [1]))
