from __future__ import annotations

import torch
from coola.equality import objects_are_equal

from batchtensor.nested import (
    cat_along_batch,
    chunk_along_batch,
    cumsum_along_batch,
    pad_along_batch,
    pad_along_seq,
    pad_sequences,
    permute_along_batch,
    select_along_batch,
    shuffle_along_batch,
    slice_along_batch,
    split_along_batch,
    sum_along_batch,
    sum_along_seq,
)
from batchtensor.tensor import lengths_to_mask


def _data() -> dict:
    return {
        "a": torch.arange(10).view(5, 2),
        "b": {"c": torch.arange(5), "d": [torch.arange(5) * 2]},
    }


def test_split_then_cat_roundtrip() -> None:
    data = _data()
    out = cat_along_batch(split_along_batch(data, split_size_or_sections=2))
    assert objects_are_equal(out, data)


def test_chunk_then_cat_roundtrip() -> None:
    data = _data()
    out = cat_along_batch(chunk_along_batch(data, chunks=3))
    assert objects_are_equal(out, data)


def test_permute_inverse_roundtrip() -> None:
    data = _data()
    perm = torch.tensor([2, 0, 4, 1, 3])
    out = permute_along_batch(permute_along_batch(data, perm), torch.argsort(perm))
    assert objects_are_equal(out, data)


def test_pad_then_slice_roundtrip() -> None:
    data = _data()
    out = slice_along_batch(pad_along_batch(data, pad_size=3), stop=5)
    assert objects_are_equal(out, data)


def test_cumsum_last_equals_sum() -> None:
    data = {"a": torch.arange(10).view(5, 2), "b": torch.arange(5)}
    last = select_along_batch(cumsum_along_batch(data), index=-1)
    assert objects_are_equal(last, sum_along_batch(data))


def test_pad_seq_and_mask_consistent() -> None:
    lengths = torch.tensor([1, 3, 2])
    mask = lengths_to_mask(lengths)
    padded = pad_along_seq({"x": torch.ones(3, 3)}, pad_size=2)
    assert isinstance(padded, dict)
    x = padded["x"]
    assert isinstance(x, torch.Tensor)
    assert x.shape[1] == mask.shape[1] + 2
    assert mask.sum(dim=1).tolist() == lengths.tolist()


def _samples() -> list:
    return [
        {"x": torch.tensor([[1.0], [2.0], [3.0]]), "ids": torch.tensor([1, 2, 3])},
        {"x": torch.tensor([[4.0]]), "ids": torch.tensor([4])},
        {"x": torch.tensor([[5.0], [6.0]]), "ids": torch.tensor([5, 6])},
    ]


def test_collate_then_mask_matches_lengths() -> None:
    samples = _samples()
    batch = pad_sequences(samples)
    mask = lengths_to_mask(torch.tensor([3, 1, 2]))
    assert objects_are_equal(sum_along_seq(batch)["ids"], torch.tensor([6, 4, 11]))
    assert objects_are_equal(batch["ids"] * mask, batch["ids"])


def test_collate_then_shuffle_keeps_the_leaves_aligned() -> None:
    batch = pad_sequences(_samples())
    out = shuffle_along_batch(batch, generator=0)
    assert objects_are_equal(out["x"].squeeze(-1), out["ids"].float())


def test_collate_then_chunk_then_cat_roundtrip() -> None:
    batch = pad_sequences(_samples())
    out = cat_along_batch(chunk_along_batch(batch, chunks=2))
    assert objects_are_equal(out, batch)
