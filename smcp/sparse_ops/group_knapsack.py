# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import List, Optional, Tuple

import torch

KnapsackGroup = Tuple[torch.Tensor, torch.Tensor]

@torch.jit.script
def _condense_tradeoff(group: KnapsackGroup, capacity: float) -> Tuple[KnapsackGroup, torch.Tensor]:
    """
    Condenses the given tradeoff group. Assumes group is total/cumulative value-cost pairs (not individual value-cost).
    """
    value, cost = group

    # Sort the value descending
    idxs = value.argsort(descending=True)
    value = value[idxs]
    cost = cost[idxs]

    mask = torch.ones_like(idxs, dtype=torch.bool)

    # Keep value only if equal to smallest cost seen so far
    # Otherwise, there is a bigger value with lower cost
    cum_mincost, _ = cost.cummin(0)
    mask.logical_and_(cum_mincost == cost)

    # Keep only the largest value for each cost
    mask[1:].logical_and_(cum_mincost[1:] != cum_mincost[:-1])

    # Keep only values meeting the capacity
    mask.logical_and_(cost <= capacity)

    # Apply current mask
    value = value[mask]
    cost = cost[mask]
    idxs = idxs[mask]
    mask = mask[mask]

    # Keep only the lowest cost for each value (in case of duplicates)
    mask[:-1].logical_and_(value[1:] != value[:-1])

    return (value[mask], cost[mask]), idxs[mask]

@torch.jit.script
def _merge_costs(
    left: torch.Tensor, right: torch.Tensor,
    merge_out: Optional[torch.Tensor] = None, keep_out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines the two tensors into a single value-cost tradeoff tensor.
    Each tensor is a tradeoff of value-cost. Each element represents the value; each position represents the cost (0, 1, 2, ..., C-1).

    Space is O(N*C). Runtime is O(N*C^2).

    Args:
        left: Tensor of value-costs. Shape (N, C)
        right: Tensor of value-costs. Shape (N, C)
        merge_out: Output storage for the merged tradeoff tensor. Shape (N, C). Can safely reuse left's storage here.
        keep_out: Output storage for the cost from the right tensor used in the merge. Shape (N, C)

    Returns:
        merge_out: New storage is created if none passed in.
        keep_out: New storage is created if none passed in.
    """
    N, C = left.shape

    if merge_out is None:
        merge_out = torch.empty_like(left)
    if keep_out is None:
        keep_out = torch.empty((N, C), dtype=torch.long, device=left.device)

    left_reversed = left.flip(1)
    for cost in range(C):
        merge_out[:, cost], keep_out[:, cost] = (right[:, :(cost+1)] + left_reversed[:, -(cost+1):]).max(dim=1)

    return merge_out, keep_out

def _group_knapsack_dp_recursive(mat: torch.Tensor, keep: torch.Tensor, costs: torch.Tensor) -> Tuple[float, int]:
    """
    Solves a group knapsack subproblem with a dynamic programming type approach.
    Mat is a tensor where each element represents the value and each position represents the cost (0, 1, 2, ..., C-1).

    Space is O(G*C). Runtime is O(G C^2) by Master theorem (root-heavy version).

    Args:
        mat: Holds the value-cost tradeoff for each group. Shape (G, C). Mutated by the method.
        keep_out: Storage for the cost of each group used in the knapsack. Shape (G, C). Mutated by the method.
        costs: Storage for the costs making up the optimal knapsack. Shape (G,). Correctly set by the method.

    Returns:
        best_value: Optimal value achieved
        best_cost: Cost at optimal value
    """
    G = mat.shape[0]

    if G == 1:
        mval, idxs = mat.max(dim=1)
        costs[0] = idxs[0]
        return mval.item(), idxs.item()

    s = G % 2
    split = G // 2 + s

    _merge_costs(mat[s:split, :], mat[split:, :], mat[s:split, :], keep[split:, :])
    best_value, best_cost = _group_knapsack_dp_recursive(mat[:split, :], keep[:split, :], costs[:split])

    costs[split:] = torch.gather(keep[split:, :], 1, costs[s:split].view(-1, 1))[:, 0]
    costs[s:split] -= costs[split:]

    return best_value, best_cost

@torch.jit.script
def _group_knapsack_dp_iterative(mat: torch.Tensor, keep: torch.Tensor, costs: torch.Tensor) -> Tuple[float, int]:
    """
    Iterative version of `_group_knapsack_dp_recursive`. Can be wrapped by torch.jit.script.
    """
    G, _ = mat.shape

    # Get subproblem sizes
    num_subproblems = torch.ceil(torch.log2(torch.tensor([G]))).item()
    subsizes = torch.ceil(G / 2 ** torch.arange(num_subproblems)).long()

    # Perform all of the merging
    end_idx = G
    for subsize in subsizes:
        end_idx = subsize.item()

        s = end_idx % 2
        split = end_idx // 2 + s
        _merge_costs(mat[s:split, :], mat[split:end_idx, :], mat[s:split, :], keep[split:end_idx, :])

    # Recover the optimal value and the constituent costs
    best_value, best_cost = mat[0,:].max(dim=0)
    costs[0] = best_cost

    for subsize in subsizes.flip(0):
        end_idx = subsize.item()

        s = end_idx % 2
        split = end_idx // 2 + s

        costs[split:end_idx] = torch.gather(keep[split:end_idx, :], 1, costs[s:split].view(-1, 1))[:, 0]
        costs[s:split] -= costs[split:end_idx]

    return best_value.item(), best_cost.item()

def _get_knapsack_matrix_from_groups(groups: List[KnapsackGroup], capacity: int, require_item: bool) -> torch.Tensor:
    """
    Creates the group knapsack tensor of shape (G, C) from the list of group (value, cost) tensors.
    Assumes each group has the values sorted descending and has unique costs.

    Args:
        groups: List of value-cost tradeoffs for each group
        capacity: Capacity of the knapsack
        require_item: Whether an item from each group must be selected

    Return:
        tradeoff: Tradeoff tensor
    """

    G = len(groups)

    if require_item:
        fill_value = float("-Inf")
    else:
        fill_value = 0

    values = torch.full((G, capacity + 2), fill_value, dtype=torch.float, device=groups[0][0].device)

    for idx, (value, cost) in enumerate(groups):
        # Clamp too costly items to one more than target; will never be used
        cost.clamp_(max=capacity + 1)

        values[idx, cost] = value

    # Remove the too costly items
    return values[:, :-1]

def _group_knapsack_dp(groups: List[KnapsackGroup], capacity: int, mode: str, require_item: bool) -> Tuple[float, int, torch.Tensor]:
    """
    Wrapper around `_group_knapsack_dp_recursive` and `_group_knapsack_dp_iterative`. Only integer costs and capacity are allowed.

    See `group_knapsack` for information on signature.
    """

    if type(capacity) != int or any(c.dtype.is_floating_point for _, c in groups):
        raise NotImplementedError("Group knapsack with a DP solver only supports integer costs and capacity")
    if any(c.min().item() < 0 for _, c in groups):
        raise NotImplementedError("Group knapsack with a DP solver only supports non-negative costs")

    mat = _get_knapsack_matrix_from_groups(groups, capacity, require_item)

    G, C = mat.shape

    keep = torch.arange(C, dtype=torch.long, device=mat.device).view(1, -1).repeat(G, 1)
    keep_costs = torch.empty(G, dtype=torch.long, device=mat.device)

    if mode == "dp":
        best_value, best_cost = _group_knapsack_dp_recursive(mat, keep, keep_costs)
    elif mode == "dp_iterative":
        best_value, best_cost = _group_knapsack_dp_iterative(mat, keep, keep_costs)
    else:
        raise NotImplementedError(f"{mode} not recognized")

    # Recover the number of items from the kept costs
    keep_idxs = torch.full_like(keep_costs, -1, dtype=torch.long)
    for gidx in range(G):
        keep_cost = keep_costs[gidx]
        _, cost = groups[gidx]

        num_item = (cost == keep_cost).nonzero()
        if num_item.numel() > 0:
            keep_idxs[gidx] = num_item[0, 0]

    return best_value, best_cost, keep_idxs

@torch.jit.script
def _merge_costs_sets(
    left: KnapsackGroup, right: KnapsackGroup,
    capacity: float
) -> Tuple[KnapsackGroup, torch.Tensor, torch.Tensor]:
    """
    Returns the merged tradeoff tensor, of size R <= N*M.
    Space is O(N*M). Runtime is O(N*M log(N*M)).

    Args:
        left: Tuple of values and corresponding costs. Each has shape (N,)
        right: Tuple of values and corresponding costs. Each has shape (M,)

    Returns:
        value: Merged tradeoff value-cost tuple. Each has shape (R,)
        left_keep: Index from the left tradeoff used in the merge. Shape (R,)
        right_keep: Index from the right tradeoff used in the merge. Shape (R,)
    """

    left_value, left_cost = left
    right_value, right_cost = right

    value = (left_value.view(-1, 1) + right_value.view(1, -1)).flatten()
    cost = (left_cost.view(-1, 1) + right_cost.view(1, -1)).flatten()

    merged, idxs = _condense_tradeoff((value, cost), capacity)

    M = right_value.shape[0]
    left_keep = torch.div(idxs, M, rounding_mode="trunc")
    right_keep = torch.fmod(idxs, M)

    return merged, left_keep, right_keep

def _group_knapsack_mim_recursive(
    groups: List[KnapsackGroup], capacity: float, keep_idxs: torch.Tensor
) -> Tuple[float, float]:
    """
    Floating-point compatible meet-in-the-middle group knapsack solver.

    In the worst-case, space is O(B^(G/2)) and runtime is O(B^(G/2) log(B)), where B is the size of the largest tradeoff set.
    If the costs are non-negative integer <= C, in the worst-case, space is O(C^2) and runtime is O(G C^2 log(C)).

    Args:
        groups: Holds the value-cost tradeoff for each group. Mutated by the method. Length G.
        capacity: Max capacity of the knapsack.
        keep_idxs: Storage for the keep_idxs making up the optimal knapsack. Shape (G,). Correctly set by the method.

    Returns:
        best_value: Optimal value achieved.
        best_cost: Cost at optimal value
    """
    G = len(groups)

    if G == 1:
        value, cost = groups[0]
        cidx = value.argmax()
        keep_idxs[0] = cidx

        return value[cidx].item(), cost[cidx].item()

    new_groups: List[KnapsackGroup] = []
    keeps: List[Tuple[torch.Tensor, torch.Tensor]] = []
    while len(groups) > 1:
        left = groups.pop(0)
        right = groups.pop(0)
        merged, left_keep, right_keep = _merge_costs_sets(left, right, capacity)

        new_groups.append(merged)
        keeps.append((left_keep, right_keep))

    new_groups.extend(groups)

    best_value, best_cost = _group_knapsack_mim_recursive(new_groups, capacity, keep_idxs[::2])

    for idx, (left_keep, right_keep) in enumerate(keeps):
        kidx = keep_idxs[2*idx]
        keep_idxs[2*idx + 1] = right_keep[kidx]
        keep_idxs[2*idx] = left_keep[kidx] # Careful: overwrites what kidx refers to

    return best_value, best_cost

def add_zero_tradeoff(group: KnapsackGroup) -> KnapsackGroup:
    # Add an artificial (0, 0) choice to the group
    v, c = group

    return (
        torch.cat([torch.tensor([0], dtype=v.dtype, device=v.device), v]),
        torch.cat([torch.tensor([0], dtype=c.dtype, device=c.device), c])
    )

def _group_knapsack_mim(groups: List[KnapsackGroup], capacity: float, require_item: bool) -> Tuple[float, float, torch.Tensor]:
    """
    Small wrapper around `_group_knapsack_mim_recursive`. Floating-point costs and capacity are allowed.

    See `group_knapsack` for information on signature.
    """
    G = len(groups)

    if require_item:
        agroups = list(groups)
    else:
        agroups = [add_zero_tradeoff(g) for g in groups]

    if G < 1:
        return -float("Inf"), float("Inf"), torch.tensor([])

    keep_idxs = torch.empty(G, dtype=torch.long, device=agroups[0][0].device)
    best_value, best_cost = _group_knapsack_mim_recursive(agroups, capacity, keep_idxs)

    if not require_item:
        keep_idxs -= 1 # Removes artificial 0 choice

    return best_value, best_cost, keep_idxs

def group_knapsack(groups: List[KnapsackGroup], capacity: float, mode: str = "mim", require_item: bool = True) -> Tuple[float, float, torch.Tensor]:
    """
    Wrapper around `_group_knapsack_dp` and `_group_knapsack_mim`.

    Args:
        groups: List of cumulative value-cost pairs for each group. Shape (G,). Not mutated.
        capacity: Knapsack capacity C.
        mode: Solver mode to use (either "dp", "dp_iterative", or "mim").
        require_item: Whether an item from each group must be selected

    Returns:
        best_value: Optimal value achieved
        best_cost: Cost at optimal value
        keep_idxs: Indexes of items from each group used. Shape (G,)
    """

    # Condense the tradeoffs (by rejecting obviously suboptimal choices)
    cgroups = []
    condense_idxs = []
    for group in groups:
        cgroup, cidxs = _condense_tradeoff(group, capacity)
        cgroups.append(cgroup)
        condense_idxs.append(cidxs)

    if mode == "dp" or mode == "dp_iterative":
        best_value, best_cost, keep_idxs = _group_knapsack_dp(cgroups, capacity, mode, require_item)
    elif mode == "mim":
        best_value, best_cost, keep_idxs = _group_knapsack_mim(cgroups, capacity, require_item)
    else:
        raise NotImplementedError(f"Group knapsack solver {mode} not recognized")

    # Adjust the number of items based on the condensed tradeoffs
    for gidx, cidxs in enumerate(condense_idxs):
        if keep_idxs[gidx] >= 0:
            keep_idxs[gidx] = cidxs[keep_idxs[gidx]]

    return best_value, best_cost, keep_idxs
