from __future__ import annotations

from batchtensor.recursive import Applier, ApplyState

################################
#     Tests for ApplyState     #
################################


def test_state_increment_depth_1() -> None:
    applier = Applier()
    assert ApplyState(applier=applier).increment_depth() == ApplyState(applier=applier, depth=1)


def test_state_increment_depth_2() -> None:
    applier = Applier()
    assert ApplyState(applier=applier).increment_depth(2) == ApplyState(applier=applier, depth=2)
