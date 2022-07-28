# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from functools import cached_property, lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, TypeVar

from tabulate import tabulate
import torch
from torch import fx, nn

from smcp.sparse_ops.shape_propagation import ShapePropagation, TensorMetadata

TElt = TypeVar("TElt")
def _group_consolidate(groups: List[Set[TElt]]) -> List[Set[TElt]]:
    # Taken from http://rosettacode.org/wiki/Set_consolidation#Python
    if len(groups) < 2:
        return [g for g in groups if len(g) > 0]
    elif len(groups[0]) == 0:
        return _group_consolidate(groups[1:])

    r, b = [groups[0]], _group_consolidate(groups[1:])
    for x in b:
        if r[0].intersection(x):
            r[0].update(x)
        elif len(x) > 0:
            r.append(x)

    return r

PruningGroup = Tuple[str,...]

class ChannelStructure:
    """
    Automatically finds the channel structure of the network, allowing to track the impacts of (input or output) channel pruning.

    NOTE: Only tested on ResNet architectures.
    """
    module_types: Dict[str, Type]
    channel_producing_modules: Tuple[nn.Module, ...] = (nn.modules.conv._ConvNd, nn.Linear)
    channel_acting_modules: Tuple[nn.Module, ...] = (nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm)

    def __init__(self, model: nn.Module, example_input: Optional[torch.Tensor] = None):
        self.module_types: Dict[str, Type] = { k: type(m) for k, m in model.named_modules() }
        self.gm = fx.symbolic_trace(model)

        if example_input is not None:
            is_training = model.training

            model.eval()
            with torch.no_grad():
                # Adds information to each node by propating the fake input
                device = next(model.parameters()).device
                example_input = example_input[:1, ...].to(device=device)
                ShapePropagation(self.gm).run(example_input)
            model.train(is_training)

    @cached_property
    def input_metadata_dict(self) -> Dict[str, TensorMetadata]:
        """
        Returns the metadata info from the input tensor to each channel-acting node.

        Assumes that channel-acting nodes take only a single input.
        """
        return {
            node.target: node.in_metadata[0]
            for node in self.gm.graph.nodes
            if hasattr(node, "in_metadata") and self._is_channel_acting(node)
        }

    @lru_cache
    def is_group_graph_cut(self, group: PruningGroup) -> bool:
        """
        Returns whether a group is a graph cut, meaning network would become untrainable if fully pruned
        """
        is_reachable: Dict[str, bool] = {}

        node: fx.Node
        for node in self.gm.graph.nodes:
            if self._is_channel_producing(node) and (node.target in group):
                is_reachable[node.name] = False
            elif len(node.all_input_nodes) == 0:
                is_reachable[node.name] = True
            else:
                is_reachable[node.name] = any(is_reachable[up.name] for up in node.all_input_nodes)

            if node.op == "output" and (not is_reachable[node.name]):
                return True

        return False

    @cached_property
    def associated_batchnorm_dict(self) -> Dict[str, str]:
        """
        Returns a mapping between 
        """
        batchnorm_dict: Dict[str, str] = {}

        node: fx.Node
        for node in self.gm.graph.nodes:
            if node.op != "call_module" or not self._is_channel_producing(node):
                continue

            for downstream_node in node.users.keys():
                if downstream_node.op != "call_module" or not issubclass(self.module_types[downstream_node.target], nn.modules.batchnorm._BatchNorm):
                    continue

                batchnorm_dict[node.target] = downstream_node.target

        return batchnorm_dict

    @cached_property
    def input_pruning_groups(self) -> List[PruningGroup]:
        """
        Returns a list of groups (of channel-producing nodes) that must be pruned together when pruning input channels.
        """
        # When pruning input channels in a given layer, we will eventually prune the corresponding output channels of the incoming layers
        # Therefore, we need to ask which layers output channels to a given layer
        # We have to do this globally, meaning if two different layers share the same input channel (from the same source) they must prune together
        # This requires (perhaps counterintuitively) us to consider the downstream nodes structure
        downstream_groups = [
            set(nlist) for nlist in self.downstream_producers_dict.values()
        ]
        groups = _group_consolidate(downstream_groups)

        # Add head nodes (i.e., aren't downstream of other nodes)
        head_groups = [
            (k,)
            for k, nlist in self.upstream_producers_dict.items()
            if len(nlist) == 0
        ]

        return head_groups + [tuple(g) for g in groups]

    @cached_property
    def output_pruning_groups(self) -> List[PruningGroup]:
        """
        Returns a list of groups (of channel-producing nodes) that must be pruned together when pruning output channels.
        """
        # (See comment on get_input_pruning_groups, but flip the logic since we're considering pruning output channels)
        upstream_groups = [
            set(nlist) for nlist in self.upstream_producers_dict.values()
        ]
        groups = _group_consolidate(upstream_groups)

        # Add tail nodes (i.e., aren't upstream of other nodes)
        tail_groups = [
            (k,)
            for k, nlist in self.downstream_producers_dict.items()
            if len(nlist) == 0
        ]

        return [tuple(g) for g in groups] + tail_groups

    @cached_property
    def upstream_producers_dict(self) -> Dict[str, List[str]]:
        """
        Returns a mapping between each channel-acting layer and the channel-producing layers upstream (i.e., that use its output channels)
        """
        return {
            str(k.target): [str(n.target) for n in nlist if self._is_channel_producing(n)]
            for k, nlist in self._upstream_nodes_dict.items()
        }

    @cached_property
    def downstream_producers_dict(self) -> Dict[str, List[str]]:
        """
        Returns a mapping between each channel-acting layer and the channel-producing layers downstream (i.e., that use its output channels)
        """
        return {
            str(k.target): [str(n.target) for n in nlist if self._is_channel_producing(n)]
            for k, nlist in self._downstream_nodes_dict.items()
        }

    def print_upstream(self) -> None:
        self._print_nodes_dict(self._upstream_nodes_dict, "inputs")

    def print_downstream(self) -> None:
        self._print_nodes_dict(self._downstream_nodes_dict, "outputs")

    def _print_nodes_dict(self, nodes_dict: Dict[fx.Node, List[fx.Node]], label: str) -> None:
        node_specs = [[n.name, [i.name for i in nodes]]
                      for n, nodes in nodes_dict.items()]
        print(tabulate(node_specs,
              headers=["name", label]))

    @cached_property
    def _upstream_nodes_dict(self) -> Dict[fx.Node, List[fx.Node]]:
        """
        Returns a mapping between each channel-acting node and the channel-acting nodes upstream (i.e., that input channels to it)
        """
        return {
            node: self._trace_channels(node, lambda n : n.all_input_nodes)
            for node in self.gm.graph.nodes
            if self._is_channel_acting(node)
        }

    @cached_property
    def _downstream_nodes_dict(self) -> Dict[fx.Node, List[fx.Node]]:
        """
        Returns a mapping between each channel-acting node and the channel-acting nodes downstream (i.e., that use its output channels)
        """
        return {
            node: self._trace_channels(node, lambda n : n.users.keys())
            for node in self.gm.graph.nodes
            if self._is_channel_acting(node)
        }

    def _trace_channels(self, node: fx.Node, next_nodes_fn: Callable[[fx.Node], Iterable[fx.Node]]) -> List[fx.Node]:
        """
        Trace the nodes along the graph (according to next_nodes_fn) to all channel-acting nodes (passing through all other nodes)
        """
        traced_nodes = []

        for next_node in next_nodes_fn(node):
            if self._is_channel_acting(next_node):
                traced_nodes.append(next_node)
            if not self._is_channel_producing(next_node):
                traced_nodes.extend(self._trace_channels(next_node, next_nodes_fn))

        return traced_nodes

    def _is_channel_producing(self, node: fx.Node) -> bool:
        """
        Returns true if the node produces its own channels
        (aka input channel size can change without output channel size changing and vice versa)
        """
        if not self._is_node_matching_module(node, self.channel_producing_modules):
            return False

        layer = self.gm.get_submodule(node.target)
        if isinstance(layer, nn.modules.conv._ConvNd):
            if layer.groups == 1:
                return True
            elif layer.groups == layer.in_channels == layer.out_channels:
                return False
            else:
                raise NotImplementedError("Channel structure has limited suppport for convolution groups")

        return True

    def _is_channel_acting(self, node: fx.Node) -> bool:
        """
        Returns true if the node's action depends on the number of channels
        """
        return self._is_node_matching_module(node, self.channel_acting_modules)

    def _is_node_matching_module(self, node: fx.Node, matches: Tuple[nn.Module, ...]) -> bool:
        return node.op == "call_module" and issubclass(self.module_types[node.target], matches)
