from __future__ import annotations

import torch
from coola.equality import objects_are_equal

from batchtensor.nested import cumprod_along_batch

#########################################
#     Tests for cumprod_along_batch     #
#########################################


def test_cumprod_along_batch_dict() -> None:
    assert objects_are_equal(
        cumprod_along_batch(
            {
                "a": torch.tensor(
                    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                ),
                "b": torch.tensor([4, 3, 2, 1, 0]),
            }
        ),
        {
            "a": torch.tensor([[1, 2], [3, 8], [15, 48], [105, 384], [945, 3840]]),
            "b": torch.tensor([4, 12, 24, 24, 0]),
        },
    )
