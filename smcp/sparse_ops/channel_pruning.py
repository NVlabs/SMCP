# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
import itertools
import math
import re
from typing import Dict, Iterable, List, Tuple, Union

import torch
from torch import nn

from smcp.sparse_ops.bn_scaling import add_bn_weight_scaling, remove_bn_weight_scaling, get_bn_scaling, set_bn_scaling
from smcp.sparse_ops.base_pruning import BasePruningMethod, PruningSchedule
from smcp.sparse_ops.channel_costing import Coster
from smcp.sparse_ops.channel_structure import ChannelStructure, PruningGroup
from smcp.sparse_ops.group_knapsack import KnapsackGroup, add_zero_tradeoff, group_knapsack
from smcp.sparse_ops.importance import ImportanceFunction
from smcp.sparse_ops.importance_accumulator import ImportanceAccumulator
from smcp.sparse_ops.parameter_masking import ParameterMaskingType, has_mask, get_mask, set_mask, get_orig
from smcp.sparse_ops.result import LayerPruneResult, NetworkPruneResult, PruneResult
from smcp.sparse_ops.scheduler import get_prune_schedule

class ChannelPruningType(Enum):
    Skip = "Skip"
    Global = "Global"

def _is_layer_masked(model: nn.Module, layer_name: str) -> bool:
    layer = model.get_submodule(layer_name)
    return has_mask(layer, "weight")

def _num_params_per_input_channel(layer: nn.Module) -> torch.Tensor:
    if isinstance(layer, nn.Conv2d):
        return layer.out_channels * math.prod(layer.kernel_size)
    elif isinstance(layer, nn.Linear):
        return layer.out_features

    raise NotImplementedError(f"Number of params per input channel not implemented for layer of type {type(layer)}")

def _get_default_output_channel_mask(layer: nn.Module) -> torch.Tensor:
    if has_mask(layer, "weight"):
        device = get_mask(layer, "weight").device
    else:
        device = layer.weight.device

    if isinstance(layer, nn.Conv2d):
        return torch.ones((layer.out_channels,), dtype=torch.bool, device=device)
    elif isinstance(layer, nn.Linear):
        return torch.ones((layer.out_features,), dtype=torch.bool, device=device)

    raise NotImplementedError(f"Default output channel mask not implemented for layer of type {type(layer)}")

def _get_effective_input_channel_mask(layer: nn.Module) -> torch.Tensor:
    if isinstance(layer, nn.Conv2d):
        if has_mask(layer, "weight"):
            return get_mask(layer, "weight").sum(dim=(2, 3)).any(dim=0)
        else:
            return torch.ones((layer.in_channels,), dtype=torch.bool, device=layer.weight.device)
    elif isinstance(layer, nn.Linear):
        if has_mask(layer, "weight"):
            return get_mask(layer, "weight").any(dim=0)
        else:
            return torch.ones((layer.in_features,), dtype=torch.bool, device=layer.weight.device)

    raise NotImplementedError(f"Input channel mask not implemented for layer of type {type(layer)}")

def _get_channel_importance_score(layer: nn.Module, importance: torch.Tensor) -> torch.Tensor:
    if isinstance(layer, nn.Conv2d):
        return importance.sum(dim=(2,3))
    elif isinstance(layer, nn.Linear):
        return importance

    raise NotImplementedError(f"Channel score not implemented for layer of type {type(layer)}")

def _get_downstream_group_dict(structure: ChannelStructure) -> Iterable[Tuple[str, PruningGroup]]:
    groups = structure.input_pruning_groups
    downstream_layer_dict = structure.downstream_producers_dict

    for layer_name, downstream_layer_names in downstream_layer_dict.items():
        if len(downstream_layer_names) == 0:
            continue

        # Find the matching group
        g = next(g for g in groups if all(n in g for n in set(downstream_layer_names)))

        yield layer_name, g

def _reverse_dict(d: Dict[str, PruningGroup]) -> Dict[PruningGroup, List[str]]:
    reversed_d = {}
    for k, v in d.items():
        reversed_d[v] = reversed_d.get(v, []) + [k]

    return reversed_d

def _is_layer_pruneable(unpruned_layers: List[Union[str, re.Pattern]], name: str) -> bool:
    for unpruned_layer in unpruned_layers:
        if isinstance(unpruned_layer, re.Pattern):
            is_match = (unpruned_layer.match(name) is not None)
        elif isinstance(unpruned_layer, str):
            is_match = (unpruned_layer == name)
        else:
            raise NotImplementedError

        if is_match:
            return False

    return True

class GroupTradeoffCalculator:
    def __init__(
        self, model: nn.Module, unpruned_layers: List[Union[str, re.Pattern]], doublesided_weight: float,
        structure: ChannelStructure, coster: Coster
    ):
        """Stateless class for calculating tradeoff importances and costs for input channel pruning"""
        self.model = model
        self.unpruned_layers = unpruned_layers
        self.doublesided_weight = doublesided_weight
        self.coster = coster

        self._groups = structure.input_pruning_groups
        self._graph_cut_groups = set(g for g in self._groups if structure.is_group_graph_cut(g))
        self._input_shape_dict = { k: m.shape[1:] for k, m in structure.input_metadata_dict.items() } # Drop batch dimension
        self._downstream_group_dict = dict(_get_downstream_group_dict(structure)) # Maps acting/producing layer -> group downstream of it (missing if no downstream)

        producing_layers = set(itertools.chain.from_iterable(self._groups))
        _actingonly_downstream_group_dict = {
            k : v for k, v in self._downstream_group_dict.items()
            if k not in producing_layers
        }
        _producer_downstream_group_dict = {
            k : v for k, v in self._downstream_group_dict.items()
            if k in producing_layers
        }

        # We assume the only channel-acting, non-producing layer that needs to be costed is a Conv2d (with groups != 1)
        _actingonly_downstream_group_dict = { 
            k : v for k, v in _actingonly_downstream_group_dict.items()
            if isinstance(self.model.get_submodule(k), (nn.Conv2d))
        }

        self._upstream_acting_layers_dict = _reverse_dict(_actingonly_downstream_group_dict) # Maps group -> acting-only layers upstream of it (missing if no upstream)
        self._upstream_producer_layers_dict = _reverse_dict(_producer_downstream_group_dict) # Maps group -> producing layers upstream of it (missing if no upstream)

    @property
    def groups(self) -> List[PruningGroup]:
        return self._groups

    def get_out_mask_dict(self, group_in_mask_dict: Dict[PruningGroup, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Returns the effective out mask for each channel producing layer (equivalent to in mask of downstream layer)
        """
        for group in group_in_mask_dict.keys():
            for n in group:
                downstream_group = self._downstream_group_dict.get(n, None)

                if downstream_group is None:
                    yield n, _get_default_output_channel_mask(self.model.get_submodule(n))
                else:
                    yield n, group_in_mask_dict[downstream_group]

    def can_prune_group(self, group: PruningGroup) -> bool:
        # Can only prune input channels if:
        # - the masking is setup
        # - there is an upstream group
        # - no layer in the group should stay unpruned
        # - no upstream group layer should stay unpruned
        return all(_is_layer_masked(self.model, n) for n in group) \
            and (group in self._upstream_producer_layers_dict) \
            and all(_is_layer_pruneable(self.unpruned_layers, n) for n in group) \
            and all(_is_layer_pruneable(self.unpruned_layers, n) for n in self._upstream_producer_layers_dict[group])

    def can_layer_prune_group(self, group: PruningGroup) -> bool:
        # Prohibit layer pruning if the group is a graph cut
        return self.can_prune_group(group) and (group not in self._graph_cut_groups)

    def get_importance(
        self, group: PruningGroup, channel_importance_dict: Dict[str, torch.Tensor], out_mask_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        group_score = 0

        # Get the score from pruning the input channels
        for layer_name in group:
            out_mask = out_mask_dict[layer_name]

            # Get the importance
            # Assumes shape (C_out, C_in)
            importance = channel_importance_dict[layer_name]

            # Get the score for each input channel
            in_score = importance[out_mask, :].sum(dim=0)

            group_score += in_score

        # Calculate abs value score of entire group
        return group_score.abs_()

    def _get_group_cost(self, group: PruningGroup, in_mask_dict: Dict[str, torch.Tensor], out_mask_dict: Dict[str, torch.Tensor], in_channels_range: List[int]) -> torch.Tensor:
        group_cost = 0

        doublesided_cost = (self.doublesided_weight != 1)

        # Determine if group can be pruned (cost calculation might not exist otherwise)
        is_pruneable = self.can_prune_group(group)

        # Get the cost from pruning the input channels in this group (all channel producers)
        for layer_name in group:
            layer = self.model.get_submodule(layer_name)

            input_shape = self._input_shape_dict[layer_name]
            in_mask = in_mask_dict[layer_name]
            out_mask = out_mask_dict[layer_name]
            eff_out_channels = out_mask.sum().item()

            if is_pruneable:
                # Get the cumulative cost to this layer of including each additional input channel
                # Assumes channel cost is order-independent (i.e., cost only depends on # of channels, not their indices)
                in_cost = torch.tensor(
                    [self.coster.get_cost(layer, input_shape, c, eff_out_channels) for c in in_channels_range],
                    device=in_mask.device, dtype=torch.float32
                )
            else:
                # Get the cost of the layer with the orig # of input channels
                in_cost = self.coster.get_cost(layer, input_shape, len(in_mask), out_mask.sum().item())
                in_cost = torch.full((len(in_channels_range),), in_cost, dtype=torch.float32, device=in_mask.device)

            # (Optionally) Downstream group gets some of the latency credit for pruning this layer's output channels
            #  which is the downstream group's input channels
            if doublesided_cost \
                and (layer_name in self._downstream_group_dict) \
                and (self.can_prune_group(self._downstream_group_dict[layer_name])):
                in_cost *= self.doublesided_weight

            group_cost += in_cost

        # Get the cost from pruning the output channels of upstream channel acting, non-producing layers
        if (group in self._upstream_acting_layers_dict):
            upstream_layers = self._upstream_acting_layers_dict[group]

            group_in_mask = in_mask_dict[group[0]]

            for layer_name in upstream_layers:
                layer = self.model.get_submodule(layer_name)

                # Conv2d is only channel acting if it has groups == in_channels == out_channels
                assert isinstance(layer, nn.Conv2d) and (layer.in_channels == layer.out_channels == layer.groups)

                input_shape = self._input_shape_dict[layer_name]

                if is_pruneable:
                    # Get the cumulative cost to this layer of including each additional channel
                    # Assumes channel cost is order-independent (i.e., cost only depends on # of channels, not their indices)
                    out_cost = torch.tensor(
                        [self.coster.get_cost(layer, input_shape, c, c) for c in in_channels_range],
                        device=group_in_mask.device, dtype=torch.float32
                    )
                else:
                    # Get the cost of the layer with the orig # of output channels
                    out_cost = self.coster.get_cost(layer, input_shape, len(group_in_mask), len(group_in_mask))
                    out_cost = torch.full((len(in_channels_range),), in_cost, dtype=torch.float32, device=group_in_mask.device)

                group_cost += out_cost

        # (Optionally) Current group gets some of the latency credit for pruning upstream producers' output channels,
        # which is this group's input channels
        if doublesided_cost and (group in self._upstream_producer_layers_dict) and is_pruneable:
            upstream_layers = self._upstream_producer_layers_dict[group]

            for layer_name in upstream_layers:
                layer = self.model.get_submodule(layer_name)

                input_shape = self._input_shape_dict[layer_name]
                in_mask = in_mask_dict[layer_name]
                eff_in_channels = in_mask.sum().item()

                # Get the cumulative cost to this layer of including each additional output channel
                # Assumes channel cost is order-independent (i.e., cost only depends on # of channels, not their indices)
                out_cost = torch.tensor(
                    [self.coster.get_cost(layer, input_shape, eff_in_channels, c) for c in in_channels_range],
                    device=in_mask.device, dtype=torch.float32
                )

                group_cost += (1 - self.doublesided_weight) * out_cost

        return group_cost

    def get_init_cost(self, group: PruningGroup) -> float:
        """
        Returns cost of an unpruned network
        """
        cost = sum(
            self.coster.get_cost(self.model.get_submodule(layer_name), self._input_shape_dict[layer_name])
            for layer_name in group
        )

        if (group in self._upstream_acting_layers_dict):
            cost += sum(
                self.coster.get_cost(self.model.get_submodule(layer_name), self._input_shape_dict[layer_name])
                for layer_name in self._upstream_acting_layers_dict[group]
            )

        return cost

    def get_current_cost(self, group: PruningGroup, in_mask_dict: Dict[str, torch.Tensor], out_mask_dict: Dict[str, torch.Tensor]) -> float:
        """
        Returns cost of current network
        """
        group_in_mask = in_mask_dict[group[0]]
        num_active_channels = group_in_mask.sum().item()

        cost = self._get_group_cost(group, in_mask_dict, out_mask_dict, [num_active_channels])

        return cost[0].item()

    def get_cum_cost(self, group: PruningGroup, in_mask_dict: Dict[str, torch.Tensor], out_mask_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns cost of varying number of group's input channels from 1 to max # channesl
        """
        group_in_mask = in_mask_dict[group[0]]

        return self._get_group_cost(group, in_mask_dict, out_mask_dict, list(range(1, len(group_in_mask) + 1)))

def _solve_global_pruning(
    group_tradeoffs: Dict[PruningGroup, KnapsackGroup], target_cost: float
) -> Tuple[float, float, Dict[PruningGroup, int]]:
    """
    Return the number of items used. Assumes each tradeoff is in increasing number of channels order.
    """
    best_value, best_cost, keep_idxs = group_knapsack(list(group_tradeoffs.values()), target_cost, mode="mim", require_item=True)

    return best_value, best_cost, { k: (n.item() + 1) for k, n in zip(group_tradeoffs.keys(), keep_idxs) }

@dataclass
class ChannelPruneSolveStats:
    curr_importance: float
    curr_cost: float
    est_importance: float
    est_cost: float
    act_cost: float

    def todict(self, prefix: str = "") -> Dict[str, str]:
        return {
            f"{prefix}importance_curr": self.curr_importance,
            f"{prefix}importance_est": self.est_importance,
            f"{prefix}cost_curr": self.curr_cost,
            f"{prefix}cost_est": self.est_cost,
            f"{prefix}cost_err": self.est_cost - self.act_cost
        }

class ChannelPruneSolver:
    def __init__(self, model: nn.Module, channel_type: ChannelPruningType, chunk_size: int, allow_layer_prune: bool, tradeoff_calc: GroupTradeoffCalculator):
        """
        Args:
            model: current network
            channel_type: type of channel pruning to perform
            chunk_size: number of consecutive channels to prune as a chunk
            allow_layer_prune: whether layer pruning via channel pruning is allowed
            tradeoff_calc: group tradeoff calculator
        """
        self.channel_type = channel_type
        self.chunk_size = chunk_size
        self.allow_layer_prune = allow_layer_prune

        self._tradeoff_calc = tradeoff_calc
        self._orig_cost = sum(tradeoff_calc.get_init_cost(g) for g in self._tradeoff_calc.groups)

        self.group_masks = {
            g: _get_effective_input_channel_mask(model.get_submodule(g[0]))
            for g in self._tradeoff_calc.groups
        }

    @property
    def group_masks(self) -> Dict[PruningGroup, torch.Tensor]:
        return self._group_in_mask

    @group_masks.setter
    def group_masks(self, group_in_mask: Dict[PruningGroup, torch.Tensor]) -> None:
        self._group_in_mask = group_in_mask
        self._in_mask_dict = { layer_name : mask for g, mask in group_in_mask.items() for layer_name in g }
        self._out_mask_dict = dict(self._tradeoff_calc.get_out_mask_dict(group_in_mask))

    def get_group_importance(self, layer_importance_dict: Dict[str, torch.Tensor]) -> Dict[PruningGroup, torch.Tensor]:
        return {
            group: self._tradeoff_calc.get_importance(group, layer_importance_dict, self._out_mask_dict)
            for group in self._tradeoff_calc.groups
        }

    def get_current_importance(self, group_importance_dict: Dict[PruningGroup, torch.Tensor]) -> float:
        """Assumes importance was calculated with the current in masks!!"""
        return sum(
            group_importance_dict[group][self._group_in_mask[group]].sum()
            for group in self._tradeoff_calc.groups
        ).item()

    def get_current_cost(self) -> float:
        return sum(
            self._tradeoff_calc.get_current_cost(group, self._in_mask_dict, self._out_mask_dict)
            for group in self._tradeoff_calc.groups
        )

    def solve(
        self, prune_ratio: float, group_importance_dict: Dict[PruningGroup, torch.Tensor]
    ) -> Tuple[Dict[PruningGroup, torch.Tensor], Tuple[float, float]]:
        """
        Get channel pruning masks.
        Assumes (and preserves) input channel mask is equal for every layer in a group.

        Args:
            prune_ratio: the target prune ratio to achieve by this epoch
            channel_importance_dict: dictionary of input channel-wise importance

        Returns:
            next_group_in_mask: new input masks for each pruning group
            solve_info: new solution's estimated importance and cost
        """
        TENSOR_CORE_MULT = 8
        assert (self.chunk_size <= TENSOR_CORE_MULT) and (TENSOR_CORE_MULT % self.chunk_size == 0), "Chunk size is incompatible with tensor core 8x efficieny"

        # Get target cost for this pruning step
        target_cost = (1 - prune_ratio) * self._orig_cost

        # Get number of channels for each group
        group_num_channels: Dict[PruningGroup, int] = { k : v.numel() for k, v in self._group_in_mask.items() }

        # Get the value-cost tradeoff for each group
        group_tradeoffs: Dict[PruningGroup, Tuple[torch.Tensor, torch.Tensor]] = {}
        group_idxs: Dict[PruningGroup, torch.Tensor] = {}
        for group in self._tradeoff_calc.groups:
            num_channels = group_num_channels[group]

            group_score = group_importance_dict[group]

            cum_group_cost = self._tradeoff_calc.get_cum_cost(group, self._in_mask_dict, self._out_mask_dict)
            idxs = torch.arange(num_channels, dtype=torch.long, device=group_score.device)

            if num_channels % self.chunk_size == 0:
                # Apply the consecutive chunk sizing
                group_score = group_score.view(-1, self.chunk_size).sum(dim=1)  # Relies on C-ordering of tensor data
                cum_group_cost = cum_group_cost[(self.chunk_size-1)::self.chunk_size]
                idxs = idxs.view(-1, self.chunk_size)
            else:
                idxs = idxs.view(-1, 1)

            # Sort score descending
            group_score, sort_idxs = group_score.sort(descending=True)
            idxs = idxs[sort_idxs, :]

            # Get cumulative score for tradeoff
            cum_group_score = group_score.cumsum(0)

            # If possible, reduce tradeoffs to multiples of 8 channels (for tensor core performance)
            # TODO: Convert to HALP latency stepping?
            if num_channels % TENSOR_CORE_MULT == 0:
                relative_mult = TENSOR_CORE_MULT // self.chunk_size
                cum_group_score = cum_group_score[(relative_mult-1)::relative_mult]
                cum_group_cost = cum_group_cost[(relative_mult-1)::relative_mult]
                idxs = idxs.view(-1, relative_mult, idxs.shape[-1])
            else:
                idxs = idxs.view(-1, 1, idxs.shape[-1])

            # Update tradeoffs based on other pruning constraints
            if not self._tradeoff_calc.can_prune_group(group):
                # Prevent any pruning for the group by limiting tradeoff to the single option using all channels
                cum_group_score = cum_group_score[-1:]
                cum_group_cost = cum_group_cost[-1:]
            elif self._tradeoff_calc.can_layer_prune_group(group) and self.allow_layer_prune:
                # Only add 0 tradeoff option if layer pruning is allowed for this group
                cum_group_score, cum_group_cost = add_zero_tradeoff((cum_group_score, cum_group_cost))

            group_tradeoffs[group] = (cum_group_score, cum_group_cost)
            group_idxs[group] = idxs # (num tradeoffs, agg size, chunk size)

        # Solve for the number of items to keep from each group
        if self.channel_type == ChannelPruningType.Global:
            est_importance, est_cost, group_num_items = _solve_global_pruning(group_tradeoffs, target_cost)
        else:
            raise NotImplementedError(f"Pruning solver not implemented for type {self.channel_type}")

        # Get the group masks
        next_group_in_mask: Dict[PruningGroup, torch.Tensor] = {}
        for group in self._tradeoff_calc.groups:
            num_channels = group_num_channels[group]
            idxs = group_idxs[group]

            # Recover number of chunks used
            num_chunks_used = group_num_items[group]
            if not self._tradeoff_calc.can_prune_group(group):
                num_chunks_used = idxs.shape[0] # All chunks were used
            elif self._tradeoff_calc.can_layer_prune_group(group) and self.allow_layer_prune:
                num_chunks_used -= 1 # First item is artifical 0 chunk

            # Set the mask
            mask = torch.zeros((num_channels), dtype=torch.bool, device=idxs.device)
            used_idxs = idxs[:num_chunks_used, ...].flatten()
            mask[used_idxs] = 1

            next_group_in_mask[group] = mask

        return next_group_in_mask, (est_importance, est_cost)

def apply_channel_pruning_mask(layer: nn.Module, channel_mask: torch.Tensor) -> LayerPruneResult:
    if isinstance(layer, nn.Conv2d):
        result = LayerPruneResult.from_masks(channel_mask, _get_effective_input_channel_mask(layer))

        set_mask(layer, "weight", channel_mask.view(1, -1, 1, 1))

        return result
    elif isinstance(layer, nn.Linear):
        result = LayerPruneResult.from_masks(channel_mask, _get_effective_input_channel_mask(layer))

        set_mask(layer, "weight", channel_mask.view(1, -1))

        return result

    raise NotImplementedError(f"Cannot apply channel mask to layer of type {type(layer)}")

class ChannelBNRescalingType(Enum):
    Skip = "Skip"
    Binary = "Binary"
    PruneFrac = "PruneFrac"
    ImportanceFrac = "ImportanceFrac"

class ChannelPruningSchedule(PruningSchedule):
    def __init__(
        self, prune_ratio: float, schedule_type: str,
        num_epochs: int, warmup_length: int, ramp_length: int, cooldown_length: int, rewiring_freq: int,
    ) -> None:
        """
        prune_ratio: fraction of neurons to prune by the end
        schedule_type: type of pruning schedule
        ramp_length: number of epochs over which to achieve desired ratio (can still rewire until the cooldown though)

        (See `PruningSchedule` for rest of args)
        """
        super().__init__(num_epochs, warmup_length, cooldown_length, rewiring_freq)

        self.prune_ratio = prune_ratio
        self.schedule_type = schedule_type
        self.ramp_length = ramp_length

        assert num_epochs >= warmup_length + ramp_length + cooldown_length, "Warmup + ramp + cooldown exceeds total number of epochs"

        epochs_at_full_ratio = num_epochs - warmup_length - ramp_length

        self._ratio_schedule = get_prune_schedule(schedule_type, prune_ratio, num_epochs, warmup_length, epochs_at_full_ratio)

    def should_prune(self, epoch: int, global_step: int) -> bool:
        return super().should_prune(epoch, global_step) and (self._ratio_schedule[epoch] > 0)

    def get_prune_ratio(self, epoch: int) -> float:
        return self._ratio_schedule[epoch].item()

class ChannelPruneResult(PruneResult):
    param_result: NetworkPruneResult
    channel_result: NetworkPruneResult
    solve_stats: ChannelPruneSolveStats

    def __init__(
        self,
        layer_param_results: Dict[str, LayerPruneResult],
        layer_channel_results: Dict[str, LayerPruneResult],
        solve_stats: ChannelPruneSolveStats
    ):
        self.param_result = NetworkPruneResult(layer_param_results)
        self.channel_result = NetworkPruneResult(layer_channel_results)
        self.solve_stats = solve_stats

    def todict(self, prefix: str = "", include_layer_info: bool = False) -> Dict[str, float]:
        return {
            **self.param_result.todict(f"{prefix}param/", include_layer_info),
            **self.channel_result.todict(f"{prefix}channel/", include_layer_info),
            **self.solve_stats.todict(f"{prefix}stats/")
        }

class ChannelPruning(BasePruningMethod):
    _schedule: ChannelPruningSchedule
    _structure: ChannelStructure
    _tradeoff_calc: GroupTradeoffCalculator
    _pruning_solver: ChannelPruneSolver

    def __init__(
        self, masking_type: ParameterMaskingType, schedule: ChannelPruningSchedule,
        importance_accumulator: ImportanceAccumulator, coster: Coster,
        channel_type: ChannelPruningType, unpruned_layers: List[Union[str, re.Pattern]], chunk_size: int, allow_layer_prune: bool,
        bn_rescaling_type: ChannelBNRescalingType, doublesided_weight: float = 1, track_mask_convergence: bool = False
    ):
        """
        Channel pruning method.

        Args:
            coster: class that gives cost of each channel
            channel_type: type of channel pruning to perform
            unpruned_layers: names of layers to keep untouched by pruning
            chunk_size: number of consecutive channels to prune as a chunk
            allow_layer_prune: whether layer pruning via channel pruning is allowed
            bn_rescaling_type: type of batchnorm rescaling to perform after pruning
            doublesided_weight: fraction of cost that should be assigned to pruning input channels (as opposed to output layers of upstream layers)

            (See `BasePruningMethod` for rest of args)
        """
        super().__init__(masking_type, schedule, importance_accumulator, track_mask_convergence)

        self.channel_type = channel_type
        self.unpruned_layers = unpruned_layers
        self.chunk_size = chunk_size
        self.allow_layer_prune = allow_layer_prune
        self.bn_rescaling_type = bn_rescaling_type
        self.doublesided_weight = doublesided_weight

        self._coster = coster

        self._structure = None
        self._tradeoff_calc = None
        self._pruning_solver = None

    def should_mask(self, name: str, layer: nn.Module) -> bool:
        return _is_layer_pruneable(self.unpruned_layers, name)

    def apply_masking(self, model: nn.Module) -> None:
        super().apply_masking(model)

        model.apply(add_bn_weight_scaling)

    def remove_masking(self, model: nn.Module) -> None:
        super().remove_masking(model)

        model.apply(remove_bn_weight_scaling)

    def update_importance(self, model: nn.Module, importance_fn: ImportanceFunction) -> None:
        if self._structure is None:
            # Initialize everything needed for channel pruning
            self._structure = ChannelStructure(model, model.example_input_array)
            self._tradeoff_calc = GroupTradeoffCalculator(
                model, self.unpruned_layers, self.doublesided_weight, self._structure, self._coster
            )
            self._pruning_solver = ChannelPruneSolver(model, self.channel_type, self.chunk_size, self.allow_layer_prune, self._tradeoff_calc)

        # Get the channel importance (Cout, Cin) for each grouplayer
        layer_importance_dict: Dict[str, torch.Tensor] = {}
        for group in self._pruning_solver.group_masks.keys():
            for layer_name in group:
                layer = model.get_submodule(layer_name)

                importance = importance_fn(get_orig(layer, "weight"))
                layer_importance_dict[layer_name] = _get_channel_importance_score(layer, importance)

        # Calculate the group importance and update the accumulator
        group_importance_dict = self._pruning_solver.get_group_importance(layer_importance_dict)

        for group, importance in group_importance_dict.items():
            self._importance_accumulator.update(group, importance)

    def prune(self, model: nn.Module, epoch: int, global_step: int) -> PruneResult:
        # Get some current information
        curr_group_in_mask = self._pruning_solver.group_masks
        curr_importance = self._pruning_solver.get_current_importance(self._importance_accumulator)
        curr_cost = self._pruning_solver.get_current_cost()

        # Solve for the next masks
        prune_ratio = self._schedule.get_prune_ratio(epoch)
        next_group_in_mask, (est_importance, est_cost) = self._pruning_solver.solve(prune_ratio, self._importance_accumulator)

        # Adjust the BN weight scaling
        if self.bn_rescaling_type != ChannelBNRescalingType.Skip:
            associated_batchnorm_dict = self._structure.associated_batchnorm_dict

            for group, new_channel_mask in next_group_in_mask.items():
                old_channel_mask = curr_group_in_mask[group]

                if torch.equal(old_channel_mask, new_channel_mask):
                    continue

                group_bn_names = list(associated_batchnorm_dict[l_n] for l_n in group if l_n in associated_batchnorm_dict)

                for bn_name in group_bn_names:
                    bn_layer: nn.modules.batchnorm._BatchNorm = model.get_submodule(bn_name)

                    if not new_channel_mask.any(): # Layer becoming fully pruned, pin as fully pruned
                        scaling = 0
                    elif self.bn_rescaling_type == ChannelBNRescalingType.Binary:
                        scaling = 1 if new_channel_mask.any() else 0
                    elif self.bn_rescaling_type == ChannelBNRescalingType.PruneFrac:
                        scaling = new_channel_mask.sum() / new_channel_mask.numel()
                    elif self.bn_rescaling_type == ChannelBNRescalingType.ImportanceFrac:
                        total_channel_importance = self._importance_accumulator[group]
                        new_channel_importance = total_channel_importance[new_channel_mask]

                        if total_channel_importance.sum() == 0:
                            # Layer had no importance, meaning it was already pruned (either directly or implicitly as part of branch)
                            # Pin as fully pruned
                            scaling = 0
                        else:
                            scaling = (new_channel_importance.sum() / total_channel_importance.sum())
                    else:
                        raise NotImplementedError(f"BN rescaling type {self.bn_rescaling_type} not recognized")

                    set_bn_scaling(bn_layer, torch.full_like(get_bn_scaling(bn_layer), scaling))

        # Apply the masks and get pruning results
        param_results_dict: Dict[str, LayerPruneResult] = {}
        channel_results_dict: Dict[str, LayerPruneResult] = {}
        for group, new_channel_mask in next_group_in_mask.items():
            old_channel_mask = curr_group_in_mask[group]

            for name in group:
                layer = model.get_submodule(name)

                # Update mask last updated tracking
                num_changed_since_start = self._track_mask_changes(global_step, name, new_channel_mask, old_channel_mask)

                # Apply the new mask
                channel_result = apply_channel_pruning_mask(layer, new_channel_mask)
                channel_result.num_changed_since_start = num_changed_since_start

                param_results_dict[name] = _num_params_per_input_channel(layer) * channel_result
                channel_results_dict[name] = channel_result

        # Update the solver's local view of the masks
        self._pruning_solver.group_masks = next_group_in_mask

        # Set the solver stats
        next_cost = self._pruning_solver.get_current_cost()
        solve_stats = ChannelPruneSolveStats(curr_importance, curr_cost, est_importance, est_cost, next_cost)

        return ChannelPruneResult(param_results_dict, channel_results_dict, solve_stats)
