r"""Compare the masked reductions with the manual ``masked_fill`` form.

``P2`` has no single torch call to match for these functions, so the
reference is the composition written by hand.
"""

from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal

from batchtensor import nested, tensor
from batchtensor.tensor import lengths_to_mask

LOWEST = torch.finfo(torch.float32).min
HIGHEST = torch.finfo(torch.float32).max


def make_tensor() -> torch.Tensor:
    return torch.arange(60, dtype=torch.float32).reshape(4, 5, 3).sin()


def make_mask() -> torch.Tensor:
    return lengths_to_mask(torch.tensor([5, 3, 1, 2]), max_len=5)


def expand(mask: torch.Tensor) -> torch.Tensor:
    return mask.unsqueeze(-1)


def manual_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~expand(mask), 0.0).sum(dim=1)


def manual_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return manual_sum(x, mask) / mask.sum(dim=1).unsqueeze(-1)


def manual_amax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~expand(mask), LOWEST).amax(dim=1)


def manual_amin(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~expand(mask), HIGHEST).amin(dim=1)


MANUAL = {"sum": manual_sum, "mean": manual_mean, "amax": manual_amax, "amin": manual_amin}
NAMES = sorted(MANUAL)


@pytest.mark.parametrize("name", NAMES)
def test_tensor_matches_manual_form(name: str) -> None:
    out = getattr(tensor, f"masked_{name}_along_seq")(make_tensor(), make_mask())
    assert objects_are_equal(out, MANUAL[name](make_tensor(), make_mask()))


@pytest.mark.parametrize("name", NAMES)
def test_nested_matches_manual_form(name: str) -> None:
    x, mask = make_tensor(), make_mask()
    out = getattr(nested, f"masked_{name}_along_seq")({"a": x, "b": [x]}, mask)
    expected = MANUAL[name](x, mask)
    assert objects_are_equal(out, {"a": expected, "b": [expected]})


@pytest.mark.parametrize("name", NAMES)
def test_masked_equals_unmasked_on_the_valid_prefix(name: str) -> None:
    x = make_tensor()
    mask = lengths_to_mask(torch.tensor([3, 3, 3, 3]), max_len=5)
    out = getattr(tensor, f"masked_{name}_along_seq")(x, mask)
    expected = getattr(tensor, f"{name}_along_seq")(x[:, :3])
    assert objects_are_allclose(out, expected)


@pytest.mark.parametrize("name", NAMES)
def test_masked_gradient_is_zero_on_padded_steps(name: str) -> None:
    x = make_tensor().requires_grad_()
    mask = make_mask()
    getattr(tensor, f"masked_{name}_along_seq")(x, mask).sum().backward()
    grad = x.grad
    assert grad is not None
    assert objects_are_equal(grad[~mask], torch.zeros(9, 3))


def test_masked_mean_pipeline_with_pad_sequences() -> None:
    samples = [
        {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        {"x": torch.tensor([[4.0]])},
        {"x": torch.tensor([[5.0], [6.0]])},
    ]
    batch = nested.pad_sequences(samples)
    mask = lengths_to_mask(torch.tensor([3, 1, 2]))
    out = nested.masked_mean_along_seq(batch, mask)
    assert objects_are_equal(out, {"x": torch.tensor([[2.0], [4.0], [5.5]])})


def test_masked_mean_pipeline_survives_shuffle_of_batch_and_mask() -> None:
    batch = {"x": make_tensor()}
    mask = make_mask()
    permutation = torch.tensor([2, 0, 3, 1])
    out = nested.masked_mean_along_seq(
        nested.permute_along_batch(batch, permutation),
        tensor.permute_along_batch(mask, permutation),
    )
    expected = nested.permute_along_batch(nested.masked_mean_along_seq(batch, mask), permutation)
    assert objects_are_equal(out, expected)
