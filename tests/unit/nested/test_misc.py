from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from batchtensor.nested import clone, contiguous, detach, pin_memory, to

###########################
#     Tests for clone     #
###########################


def test_clone_tensor() -> None:
    tensor = torch.tensor([[0, 1], [2, 3]])
    out = clone(tensor)
    assert objects_are_equal(out, tensor)
    assert out is not tensor
    out[0, 0] = 999
    assert tensor[0, 0].item() == 0


def test_clone_dict() -> None:
    data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
    out = clone(data)
    assert objects_are_equal(out, data)
    assert out["a"] is not data["a"]


################################
#     Tests for contiguous     #
################################


def test_contiguous_tensor() -> None:
    tensor = torch.tensor([[0, 1], [2, 3]]).t()
    out = contiguous(tensor)
    assert objects_are_equal(out, tensor)
    assert out.is_contiguous()


def test_contiguous_dict() -> None:
    data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
    out = contiguous(data)
    assert objects_are_equal(out, data)


############################
#     Tests for detach     #
############################


def test_detach_tensor() -> None:
    tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    out = detach(tensor)
    assert out.requires_grad is False
    assert objects_are_equal(out, tensor.detach())


def test_detach_dict() -> None:
    data = {
        "a": torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True),
        "b": torch.tensor([4.0, 5.0]),
    }
    out = detach(data)
    assert out["a"].requires_grad is False


################################
#     Tests for pin_memory     #
################################


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_tensor() -> None:
    tensor = torch.tensor([[0, 1], [2, 3]])
    out = pin_memory(tensor)
    assert objects_are_equal(out, tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_dict() -> None:
    data = {"a": torch.tensor([[0, 1], [2, 3]]), "b": torch.tensor([4, 5])}
    out = pin_memory(data)
    assert objects_are_equal(out, data)


########################
#     Tests for to     #
########################


def test_to_dtype_tensor() -> None:
    assert objects_are_equal(
        to(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), dtype=torch.float),
        torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
    )


def test_to_dtype_dict() -> None:
    assert objects_are_equal(
        to(
            {
                "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": torch.tensor([4, 3, 2, 1, 0]),
            },
            dtype=torch.float,
        ),
        {
            "a": torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
            "b": torch.tensor([4, 3, 2, 1, 0], dtype=torch.float),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_to_device(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        to(
            {
                "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": torch.tensor([4, 3, 2, 1, 0]),
            },
            device=device,
        ),
        {
            "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], device=device),
            "b": torch.tensor([4, 3, 2, 1, 0], device=device),
        },
    )
