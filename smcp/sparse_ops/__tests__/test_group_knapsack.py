# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import List, Tuple

import pytest
import torch

from smcp.sparse_ops.group_knapsack import (
    group_knapsack, _condense_tradeoff, _get_knapsack_matrix_from_groups,
    _merge_costs, _merge_costs_sets, KnapsackGroup
)

def create_mock_groups(G: int = 64, max_items_per_group: int = 14) -> Tuple[List[KnapsackGroup], int]:
    group_values = 10 * torch.rand((G, max_items_per_group)) - 2
    group_costs = torch.randint(0, 15, (G, max_items_per_group), dtype=torch.long)

    num_items_per_group = torch.randint(3, max_items_per_group, (G,))

    groups: List[KnapsackGroup] = []
    for gidx in range(G):
        vals = group_values[gidx, :num_items_per_group[gidx]]
        costs = group_costs[gidx, :num_items_per_group[gidx]]

        vals, indexes = vals.sort(dim=0, descending=True)
        costs = costs[indexes]

        vals.cumsum_(0)
        costs.cumsum_(0)

        groups.append((vals, costs))

    min_cost = sum(c[0].item() for _, c in groups)
    max_cost = sum(c[-1].item() for _, c in groups)
    target_cost = int((min_cost + max_cost)*0.5)

    return groups, target_cost

def test_condense_tradeoff():
    groups, capacity = create_mock_groups(1)
    group = groups[0]

    (cvalue, ccost), idxs = _condense_tradeoff(group, capacity)

    assert torch.equal(group[0][idxs], cvalue), "Idxs does not match condensed value"
    assert torch.equal(group[1][idxs], ccost), "Idxs does not match condensed cost"

    assert torch.all(cvalue[:-1] > cvalue[1:]), "Condensed value is not sorted descending and unique"
    assert torch.all(ccost[:-1] > ccost[1:]), "Condensed cost is not sorted descending and unique"

@pytest.mark.parametrize("require_item", [True, False])
def test_merge_costs(require_item: bool):
    groups, target_cost = create_mock_groups(G=2)
    mat = _get_knapsack_matrix_from_groups(groups, target_cost, require_item)
    _, C = mat.shape

    merged, right_keep = _merge_costs(mat[:1], mat[1:])

    left_keep = torch.arange(C).view(1, -1) - right_keep
    left_contrib_values = torch.gather(mat[:1], 1, left_keep.long())
    right_contrib_values = torch.gather(mat[1:], 1, right_keep.long())

    assert torch.equal(left_contrib_values + right_contrib_values, merged), "Keep and merged result are not in sync"

def test_merge_costs_sets():
    groups, target_cost = create_mock_groups(2)

    left = groups[0]
    right = groups[1]
    (value, cost), left_keep, right_keep = _merge_costs_sets(left, right, target_cost)

    assert torch.equal(value.sort(descending=True)[0], value), "Value tradeoff is not sorted descending"
    assert torch.equal(cost.sort(descending=True)[0], cost), "Cost tradeoff is not sorted descending"
    assert torch.equal(left[0][left_keep] + right[0][right_keep], value), "Keep and tradeoff value are not in sync"
    assert torch.equal(left[1][left_keep] + right[1][right_keep], cost), "Keep and tradeoff cost are not in sync"

    assert cost[0].item() <= target_cost, "Most costly item exceeds the capacity"

knapsack_modes = ["dp", "dp_iterative", "mim"]

@pytest.mark.parametrize("G", [2, 64, 46])
@pytest.mark.parametrize("mode", knapsack_modes)
@pytest.mark.parametrize("require_item", [True, False])
def test_group_knapsack(G: int, mode: str, require_item: bool):
    groups, target_cost = create_mock_groups(G)

    best_value, best_cost, keep_idxs = group_knapsack(groups, target_cost, mode, require_item)

    assert len(keep_idxs) == G, "Wrong number of items returned"

    items_value = sum([
        group[0][kidx]
        for group, kidx in zip(groups, keep_idxs)
        if kidx >= 0
    ])
    items_cost = sum([
        group[1][kidx]
        for group, kidx in zip(groups, keep_idxs)
        if kidx >= 0
    ])

    assert items_cost == best_cost, "Chosen items do not equal the best cost"
    assert best_cost <= target_cost, "Items cost more than the target"

    assert items_value == pytest.approx(best_value), "Chosen items do not equal the best value"

    if require_item:
        assert torch.all(keep_idxs >= 0), "A group is not used despite an item being required"

@pytest.mark.parametrize("G", [2, 64, 46])
@pytest.mark.parametrize("require_item", [True, False])
def test_group_knapsack_methods_equivalent(G: int, require_item: bool):
    groups, target_cost = create_mock_groups(G)

    best_value, best_cost, keep_idxs = group_knapsack(groups, target_cost, knapsack_modes[0], require_item)

    for mode in knapsack_modes[1:]:
        v, c, it = group_knapsack(groups, target_cost, mode, require_item)

        assert best_value == pytest.approx(v), "Best values do not agree"
        assert best_cost == pytest.approx(c), "Best costs do not agree"
        assert torch.equal(keep_idxs, it), "Selected items do not agree"
