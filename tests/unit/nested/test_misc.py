from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from batchtensor.nested import to

########################
#     Tests for to     #
########################


def test_to_dtype_tensor() -> None:
    assert objects_are_equal(
        to(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), dtype=torch.float),
        torch.arange(10, dtype=torch.float).view(5, 2),
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
            "a": torch.arange(10, dtype=torch.float).view(5, 2),
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
            "a": torch.arange(10, device=device).view(5, 2),
            "b": torch.tensor([4, 3, 2, 1, 0], device=device),
        },
    )
