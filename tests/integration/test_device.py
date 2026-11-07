from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from batchtensor import nested, tensor

ACCELERATORS = [device for device in get_available_devices() if device != "cpu"]


@pytest.mark.parametrize("device", ACCELERATORS)
def test_shuffle_keeps_the_device(device: str) -> None:
    out = nested.shuffle_along_batch({"a": torch.arange(4, device=device)}, generator=0)
    assert out["a"].device.type == torch.device(device).type


@pytest.mark.parametrize("device", ACCELERATORS)
def test_shuffle_with_a_cpu_generator(device: str) -> None:
    x = torch.arange(4, device=device)
    expected = tensor.shuffle_along_batch(torch.arange(4), generator=0)
    assert objects_are_equal(tensor.shuffle_along_batch(x, generator=0).cpu(), expected)


@pytest.mark.parametrize("device", ACCELERATORS)
def test_index_select_keeps_the_device(device: str) -> None:
    x = torch.arange(6, device=device).reshape(2, 3)
    out = tensor.index_select_along_seq(x, torch.tensor([[0, 1], [2, 0]], device=device))
    assert out.device.type == torch.device(device).type


@pytest.mark.parametrize("device", ACCELERATORS)
def test_lengths_to_mask_keeps_the_device(device: str) -> None:
    assert tensor.lengths_to_mask(torch.tensor([1, 2], device=device)).device.type == (
        torch.device(device).type
    )
