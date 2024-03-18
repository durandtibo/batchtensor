from __future__ import annotations

from unittest.mock import Mock

import torch
from coola import objects_are_equal
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from batchtensor.nested import from_numpy

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

################################
#     Tests for from_numpy     #
################################


@numpy_available
def test_from_numpy_array_float32() -> None:
    assert objects_are_equal(from_numpy(np.ones((2, 3), dtype=np.float32)), torch.ones(2, 3))


@numpy_available
def test_from_numpy_array_float64() -> None:
    assert objects_are_equal(from_numpy(np.ones((2, 3))), torch.ones(2, 3, dtype=torch.float64))


@numpy_available
def test_from_numpy_dict() -> None:
    assert objects_are_equal(
        from_numpy(
            {
                "a": np.array(
                    [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], dtype=np.float32
                ),
                "b": np.array([4, 3, 2, 1, 0], dtype=int),
            }
        ),
        {
            "a": torch.tensor(
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], dtype=torch.float32
            ),
            "b": torch.tensor([4, 3, 2, 1, 0], dtype=torch.long),
        },
    )


@numpy_available
def test_from_numpy_nested() -> None:
    assert objects_are_equal(
        from_numpy(
            {
                "a": np.array(
                    [[4.0, 9.0], [1.0, 7.0], [2.0, 5.0], [5.0, 6.0], [3.0, 8.0]], dtype=np.float64
                ),
                "b": np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32),
                "list": [np.array([5, 6, 7, 8, 9], dtype=int)],
            }
        ),
        {
            "a": torch.tensor(
                [[4.0, 9.0], [1.0, 7.0], [2.0, 5.0], [5.0, 6.0], [3.0, 8.0]], dtype=torch.float64
            ),
            "b": torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.float),
            "list": [torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)],
        },
    )
