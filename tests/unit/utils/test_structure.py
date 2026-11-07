from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch
from coola.equality import objects_are_equal

from batchtensor.utils.structure import (
    _check_tensor,
    _guard,
    _Parts,
    apply_to_tensors,
    combine_tensors,
    get_size,
    split_tensors,
)


class Point(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class Pair:
    first: torch.Tensor
    second: torch.Tensor


def _zeros() -> torch.Tensor:
    return torch.zeros(1)


def cat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=0)


def chunk2(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tensor.chunk(2, dim=0)


######################################
#     Tests for apply_to_tensors     #
######################################


def test_apply_to_tensors_tensor() -> None:
    assert objects_are_equal(apply_to_tensors(torch.ones(2), torch.neg), -torch.ones(2))


def test_apply_to_tensors_dict() -> None:
    out = apply_to_tensors({"a": torch.ones(2), "b": torch.zeros(1)}, lambda t: t + 1)
    assert objects_are_equal(out, {"a": torch.full((2,), 2.0), "b": torch.ones(1)})


def test_apply_to_tensors_list() -> None:
    assert objects_are_equal(
        apply_to_tensors([torch.ones(2), torch.ones(1)], torch.neg),
        [-torch.ones(2), -torch.ones(1)],
    )


def test_apply_to_tensors_tuple() -> None:
    assert objects_are_equal(apply_to_tensors((torch.ones(2),), torch.neg), (-torch.ones(2),))


def test_apply_to_tensors_namedtuple() -> None:
    out = apply_to_tensors(Point(torch.ones(1), torch.ones(1)), torch.neg)
    assert objects_are_equal(out, Point(-torch.ones(1), -torch.ones(1)))


def test_apply_to_tensors_ordered_dict() -> None:
    out = apply_to_tensors(OrderedDict(b=torch.ones(1), a=torch.ones(1)), torch.neg)
    assert objects_are_equal(out, OrderedDict(b=-torch.ones(1), a=-torch.ones(1)))


def test_apply_to_tensors_defaultdict() -> None:
    out = apply_to_tensors(defaultdict(_zeros, {"a": torch.ones(1)}), torch.neg)
    assert objects_are_equal(out, defaultdict(_zeros, {"a": -torch.ones(1)}))


def test_apply_to_tensors_dataclass() -> None:
    out = apply_to_tensors(Pair(torch.ones(1), torch.zeros(1)), lambda t: t + 1)
    assert objects_are_equal(out, Pair(torch.full((1,), 2.0), torch.ones(1)))


def test_apply_to_tensors_nested_mix() -> None:
    data = {"a": [torch.ones(1), (torch.zeros(1),)], "b": {"c": torch.ones(2)}}
    out = apply_to_tensors(data, lambda t: t + 1)
    assert objects_are_equal(
        out,
        {"a": [torch.full((1,), 2.0), (torch.ones(1),)], "b": {"c": torch.full((2,), 2.0)}},
    )


def test_apply_to_tensors_empty_dict() -> None:
    assert apply_to_tensors({}, torch.neg) == {}


def test_apply_to_tensors_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        apply_to_tensors({"a": torch.ones(1), "b": "abc"}, torch.neg)


def test_apply_to_tensors_non_tensor_leaf_in_list() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received NoneType"):
        apply_to_tensors([None], torch.neg)


def test_apply_to_tensors_does_not_mutate_input() -> None:
    data = {"a": torch.ones(2)}
    apply_to_tensors(data, torch.neg)
    assert objects_are_equal(data, {"a": torch.ones(2)})


def test_apply_to_tensors_repeat_call() -> None:
    data = {"a": torch.ones(2)}
    assert objects_are_equal(apply_to_tensors(data, torch.neg), apply_to_tensors(data, torch.neg))


def test_apply_to_tensors_fn_can_return_non_tensor() -> None:
    out = apply_to_tensors({"a": torch.ones(2, 3)}, lambda t: t.max(dim=0))
    assert isinstance(out["a"], tuple)


###################################
#     Tests for split_tensors     #
###################################


def test_split_tensors_tensor() -> None:
    assert objects_are_equal(
        split_tensors(torch.arange(4), chunk2), (torch.tensor([0, 1]), torch.tensor([2, 3]))
    )


def test_split_tensors_dict() -> None:
    out = split_tensors({"a": torch.arange(4), "b": torch.arange(4) + 10}, chunk2)
    assert objects_are_equal(
        out,
        (
            {"a": torch.tensor([0, 1]), "b": torch.tensor([10, 11])},
            {"a": torch.tensor([2, 3]), "b": torch.tensor([12, 13])},
        ),
    )


def test_split_tensors_list() -> None:
    out = split_tensors([torch.arange(4)], chunk2)
    assert objects_are_equal(out, ([torch.tensor([0, 1])], [torch.tensor([2, 3])]))


def test_split_tensors_tuple() -> None:
    out = split_tensors((torch.arange(4),), chunk2)
    assert objects_are_equal(out, ((torch.tensor([0, 1]),), (torch.tensor([2, 3]),)))


def test_split_tensors_namedtuple() -> None:
    out = split_tensors(Point(torch.arange(4), torch.arange(4)), chunk2)
    assert objects_are_equal(
        out,
        (
            Point(torch.tensor([0, 1]), torch.tensor([0, 1])),
            Point(torch.tensor([2, 3]), torch.tensor([2, 3])),
        ),
    )


def test_split_tensors_ordered_dict() -> None:
    out = split_tensors(OrderedDict(b=torch.arange(4), a=torch.arange(4)), chunk2)
    assert type(out[0]) is OrderedDict


def test_split_tensors_nested_mix() -> None:
    out = split_tensors({"a": torch.arange(4), "b": [torch.arange(4), (torch.arange(4),)]}, chunk2)
    assert objects_are_equal(
        out,
        (
            {"a": torch.tensor([0, 1]), "b": [torch.tensor([0, 1]), (torch.tensor([0, 1]),)]},
            {"a": torch.tensor([2, 3]), "b": [torch.tensor([2, 3]), (torch.tensor([2, 3]),)]},
        ),
    )


def test_split_tensors_single_part() -> None:
    out = split_tensors({"a": torch.arange(2)}, lambda t: (t,))
    assert objects_are_equal(out, ({"a": torch.tensor([0, 1])},))


def test_split_tensors_empty_dict() -> None:
    with pytest.raises(ValueError, match="does not contain any tensor"):
        split_tensors({}, chunk2)


def test_split_tensors_zero_part() -> None:
    with pytest.raises(ValueError, match="at least one piece"):
        split_tensors({"a": torch.arange(2)}, lambda _: ())


def test_split_tensors_different_part_counts() -> None:
    with pytest.raises(ValueError, match=r"same number of pieces but got \[2, 3\]"):
        split_tensors(
            {"a": torch.arange(4), "b": torch.arange(6)},
            lambda t: t.chunk(2 if t.shape[0] == 4 else 3, dim=0),
        )


def test_split_tensors_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received int"):
        split_tensors({"a": torch.arange(4), "b": 1}, chunk2)


def test_split_tensors_does_not_mutate_input() -> None:
    data = {"a": torch.arange(4)}
    split_tensors(data, chunk2)
    assert objects_are_equal(data, {"a": torch.arange(4)})


def test_split_tensors_repeat_call() -> None:
    data = {"a": torch.arange(4)}
    assert objects_are_equal(split_tensors(data, chunk2), split_tensors(data, chunk2))


#####################################
#     Tests for combine_tensors     #
#####################################


def test_combine_tensors_tensor() -> None:
    assert objects_are_equal(
        combine_tensors([torch.ones(1), torch.zeros(2)], cat), torch.tensor([1.0, 0.0, 0.0])
    )


def test_combine_tensors_dict() -> None:
    out = combine_tensors([{"a": torch.ones(1)}, {"a": torch.zeros(1)}], cat)
    assert objects_are_equal(out, {"a": torch.tensor([1.0, 0.0])})


def test_combine_tensors_dict_different_key_order() -> None:
    out = combine_tensors(
        [{"a": torch.ones(1), "b": torch.ones(1)}, {"b": torch.zeros(1), "a": torch.zeros(1)}], cat
    )
    assert objects_are_equal(out, {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 0.0])})


def test_combine_tensors_list() -> None:
    out = combine_tensors([[torch.ones(1)], [torch.zeros(1)]], cat)
    assert objects_are_equal(out, [torch.tensor([1.0, 0.0])])


def test_combine_tensors_tuple() -> None:
    out = combine_tensors([(torch.ones(1),), (torch.zeros(1),)], cat)
    assert objects_are_equal(out, (torch.tensor([1.0, 0.0]),))


def test_combine_tensors_namedtuple() -> None:
    items = [Point(torch.ones(1), torch.ones(1)), Point(torch.zeros(1), torch.zeros(1))]
    out = combine_tensors(items, cat)
    assert objects_are_equal(out, Point(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])))


def test_combine_tensors_ordered_dict() -> None:
    out = combine_tensors([OrderedDict(a=torch.ones(1))] * 2, cat)
    assert type(out) is OrderedDict


def test_combine_tensors_defaultdict() -> None:
    out = combine_tensors([defaultdict(_zeros, {"a": torch.ones(1)})] * 2, cat)
    assert objects_are_equal(out, defaultdict(_zeros, {"a": torch.ones(2)}))


def test_combine_tensors_nested_mix() -> None:
    items = [
        {"a": torch.ones(1), "b": [torch.ones(1), (torch.ones(1),)]},
        {"a": torch.zeros(1), "b": [torch.zeros(1), (torch.zeros(1),)]},
    ]
    assert objects_are_equal(
        combine_tensors(items, cat),
        {
            "a": torch.tensor([1.0, 0.0]),
            "b": [torch.tensor([1.0, 0.0]), (torch.tensor([1.0, 0.0]),)],
        },
    )


def test_combine_tensors_single_item() -> None:
    assert objects_are_equal(combine_tensors([{"a": torch.ones(1)}], cat), {"a": torch.ones(1)})


def test_combine_tensors_empty_items() -> None:
    with pytest.raises(ValueError, match="empty sequence of structures"):
        combine_tensors([], cat)


def test_combine_tensors_different_keys() -> None:
    with pytest.raises(ValueError, match="must have the same keys"):
        combine_tensors([{"a": torch.ones(1)}, {"b": torch.ones(1)}], cat)


def test_combine_tensors_mapping_and_tensor() -> None:
    with pytest.raises(ValueError, match="item 1 has keys Tensor"):
        combine_tensors([{"a": torch.ones(1)}, torch.ones(1)], cat)


def test_combine_tensors_different_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        combine_tensors([[torch.ones(1)], [torch.ones(1), torch.ones(1)]], cat)


def test_combine_tensors_list_and_tensor() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        combine_tensors([[torch.ones(1)], torch.ones(1)], cat)


def test_combine_tensors_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received str"):
        combine_tensors([{"a": "x"}, {"a": "y"}], cat)


def test_combine_tensors_does_not_mutate_input() -> None:
    items = [{"a": torch.ones(1)}, {"a": torch.zeros(1)}]
    combine_tensors(items, cat)
    assert objects_are_equal(items, [{"a": torch.ones(1)}, {"a": torch.zeros(1)}])


def test_combine_tensors_repeat_call() -> None:
    items = [{"a": torch.ones(1)}, {"a": torch.zeros(1)}]
    assert objects_are_equal(combine_tensors(items, cat), combine_tensors(items, cat))


##############################
#     Tests for get_size     #
##############################


def test_get_size_tensor() -> None:
    assert get_size(torch.ones(3, 2), dim=0) == 3


def test_get_size_dim_1() -> None:
    assert get_size({"a": torch.ones(3, 2), "b": [torch.ones(5, 2)]}, dim=1) == 2


def test_get_size_nested() -> None:
    assert get_size({"a": torch.ones(3, 2), "b": [torch.ones(3), (torch.ones(3, 4),)]}, dim=0) == 3


def test_get_size_no_leaf() -> None:
    with pytest.raises(ValueError, match="does not contain any tensor"):
        get_size({}, dim=0)


def test_get_size_different_sizes() -> None:
    with pytest.raises(ValueError, match=r"same size along dimension 0 but got \[2, 3\]"):
        get_size({"a": torch.ones(2), "b": torch.ones(3)}, dim=0)


def test_get_size_zero_dim_tensor() -> None:
    with pytest.raises(ValueError, match=r"at least 1 dimension\(s\) but got shape \(\)"):
        get_size(torch.tensor(1.0), dim=0)


def test_get_size_missing_dim() -> None:
    with pytest.raises(ValueError, match=r"at least 2 dimension\(s\) but got shape \(3,\)"):
        get_size(torch.ones(3), dim=1)


def test_get_size_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received int"):
        get_size({"a": 1}, dim=0)


###################################
#     Tests for _check_tensor     #
###################################


def test_check_tensor_tensor() -> None:
    assert _check_tensor(torch.ones(1)) is None


def test_check_tensor_non_tensor() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received list"):
        _check_tensor([1])


############################
#     Tests for _guard     #
############################


def test_guard_tensor() -> None:
    assert objects_are_equal(_guard(torch.neg)(torch.ones(1)), -torch.ones(1))


def test_guard_non_tensor() -> None:
    with pytest.raises(TypeError, match=r"Expected a torch.Tensor leaf but received float"):
        _guard(torch.neg)(1.0)


############################
#     Tests for _Parts     #
############################


def test_parts_keeps_parts() -> None:
    parts = (torch.ones(1),)
    assert _Parts(parts).parts is parts
