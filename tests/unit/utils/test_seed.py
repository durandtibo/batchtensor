from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.random import make_torch_generator

from batchtensor.utils.seed import random_permutation, setup_torch_generator

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


########################################
#     Tests for random_permutation     #
########################################


def test_random_permutation_is_permutation() -> None:
    assert objects_are_equal(random_permutation(5).sort().values, torch.arange(5))


def test_random_permutation_dtype() -> None:
    assert random_permutation(5).dtype == torch.long


def test_random_permutation_size_zero() -> None:
    assert objects_are_equal(random_permutation(0), torch.zeros(0, dtype=torch.long))


def test_random_permutation_generator() -> None:
    assert objects_are_equal(
        random_permutation(10, generator=make_torch_generator(1)),
        random_permutation(10, generator=make_torch_generator(1)),
    )


def test_random_permutation_seed_same_as_generator() -> None:
    assert objects_are_equal(
        random_permutation(10, generator=1),
        random_permutation(10, generator=make_torch_generator(1)),
    )


def test_random_permutation_seed_value() -> None:
    assert objects_are_equal(
        random_permutation(5, generator=0), torch.randperm(5, generator=make_torch_generator(0))
    )


def test_random_permutation_none_uses_global_rng() -> None:
    torch.manual_seed(0)
    expected = torch.randperm(10)
    torch.manual_seed(0)
    assert objects_are_equal(random_permutation(10), expected)


def test_random_permutation_generator_state_advances() -> None:
    generator = make_torch_generator(1)
    assert not objects_are_equal(
        random_permutation(10, generator=generator), random_permutation(10, generator=generator)
    )


def test_random_permutation_negative_size() -> None:
    with pytest.raises(RuntimeError, match="negative dimension"):
        random_permutation(-1)
