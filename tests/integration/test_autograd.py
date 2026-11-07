from __future__ import annotations

import torch
from coola.equality import objects_are_equal

from batchtensor import nested


def test_gradient_flows_through_a_nested_chain() -> None:
    x = torch.ones(4, 3, requires_grad=True)
    y = torch.ones(4, 3, requires_grad=True)
    data = {"x": x, "y": y}
    out = nested.mean_along_seq(nested.cumsum_along_batch(nested.slice_along_batch(data, stop=2)))
    (out["x"].sum() + out["y"].sum()).backward()
    assert objects_are_equal(x.grad, y.grad)


def test_gradient_values_through_a_nested_chain() -> None:
    x = torch.ones(4, 3, requires_grad=True)
    out = nested.mean_along_seq(nested.cumsum_along_batch(nested.slice_along_batch(x, stop=2)))
    out.sum().backward()
    assert objects_are_equal(
        x.grad,
        torch.tensor(
            [[0.6666666666666666] * 3, [0.3333333333333333] * 3, [0.0] * 3, [0.0] * 3]
        ).float(),
    )


def test_detach_removes_the_gradient() -> None:
    assert nested.detach(torch.ones(2, requires_grad=True)).requires_grad is False
