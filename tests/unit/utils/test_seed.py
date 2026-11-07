from __future__ import annotations

from coola.equality import objects_are_equal
from coola.random import make_torch_generator

from batchtensor.utils.seed import setup_torch_generator

###########################################
#     Tests for setup_torch_generator     #
###########################################


def test_setup_torch_generator() -> None:
    generator = make_torch_generator(1)
    assert setup_torch_generator(generator) is generator


def test_setup_torch_generator_seed() -> None:
    assert objects_are_equal(
        setup_torch_generator(1).get_state(), make_torch_generator(1).get_state()
    )
