from __future__ import annotations

import logging

import torch
from coola.equality import objects_are_equal

from batchtensor import nested, tensor
from batchtensor.utils.structure import apply_to_tensors

logger = logging.getLogger(__name__)


def check_nested() -> None:
    logger.info("Checking batchtensor.nested package...")

    assert objects_are_equal(
        nested.index_select_along_batch(
            {
                "a": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": torch.tensor([[5], [4], [3], [2], [1]]),
            },
            torch.tensor([4, 3, 2, 1, 0]),
        ),
        {
            "a": torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
            "b": torch.tensor([[1], [2], [3], [4], [5]]),
        },
    )


def check_utils() -> None:
    logger.info("Checking batchtensor.utils package...")

    assert objects_are_equal(
        apply_to_tensors({"a": torch.ones(2), "b": [torch.zeros(1)]}, torch.neg),
        {"a": -torch.ones(2), "b": [-torch.zeros(1)]},
    )


def check_tensor() -> None:
    logger.info("Checking batchtensor.tensor package...")

    assert objects_are_equal(
        tensor.index_select_along_batch(
            torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), torch.tensor([4, 3, 2, 1, 0])
        ),
        torch.tensor([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


def main() -> None:
    check_nested()
    check_tensor()
    check_utils()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
