# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from copy import deepcopy
from typing import Any, Optional, Set

import torch
from torch import fx, nn

from smcp.sparse_ops.channel_structure import ChannelStructure, PruningGroup, TensorMetadata

def _get_input_mask(layer: nn.Module) -> Optional[torch.Tensor]:
    if isinstance(layer, nn.modules.conv._ConvNd):
        return (layer.weight != 0).flatten(start_dim=2).any(dim=-1).any(dim=0)
    elif isinstance(layer, nn.Linear):
        return (layer.weight != 0).any(dim=0)

    return None

def _slim_layer_input(layer: nn.Module, in_mask: torch.Tensor) -> None:
    new_in_channels = in_mask.sum().item()
    if isinstance(layer, nn.modules.conv._ConvNd):
        assert layer.groups == 1 or layer.groups == layer.in_channels == layer.out_channels

        layer.in_channels = new_in_channels
        if layer.groups == 1:
            layer.weight.data = layer.weight[:, in_mask, ...]
        else:
            layer.groups = new_in_channels
            layer.out_channels = new_in_channels
            layer.weight.data = layer.weight[in_mask, :, ...]
    elif isinstance(layer, nn.Linear):
        layer.in_features = new_in_channels
        layer.weight.data = layer.weight[:, in_mask]
    elif isinstance(layer, nn.modules.batchnorm._BatchNorm):
        layer.num_features = new_in_channels
        if layer.weight is not None:
            layer.weight.data = layer.weight[in_mask]
        if layer.bias is not None:
            layer.bias.data = layer.bias[in_mask]
        if layer.running_mean is not None:
            layer.running_mean.data = layer.running_mean[in_mask]
        if layer.running_var is not None:
            layer.running_var.data = layer.running_var[in_mask]

def _get_output_mask(layer: nn.Module) -> Optional[torch.Tensor]:
    if isinstance(layer, nn.modules.conv._ConvNd):
        return (layer.weight != 0).flatten(start_dim=1).any(dim=1)
    elif isinstance(layer, nn.Linear):
        return (layer.weight != 0).any(dim=1)

    return None

def _slim_layer_output(layer: nn.Module, out_mask: torch.Tensor) -> None:
    new_out_channels = out_mask.sum().item()
    if isinstance(layer, nn.modules.conv._ConvNd):
        assert layer.groups == 1 or layer.groups == layer.in_channels == layer.out_channels

        if layer.groups != 1:
            layer.groups = new_out_channels
            layer.in_channels = new_out_channels

        layer.out_channels = new_out_channels
        layer.weight.data = layer.weight[out_mask, ...]
        if layer.bias is not None:
            layer.bias.data = layer.bias[out_mask]
    elif isinstance(layer, nn.Linear):
        layer.out_features = new_out_channels
        layer.weight.data = layer.weight[out_mask, :]
        if layer.bias is not None:
            layer.bias.data = layer.bias[out_mask]
    elif isinstance(layer, nn.modules.batchnorm._BatchNorm):
        layer.num_features = new_out_channels
        if layer.weight is not None:
            layer.weight.data = layer.weight[out_mask]
        if layer.bias is not None:
            layer.bias.data = layer.bias[out_mask]
        if layer.running_mean is not None:
            layer.running_mean.data = layer.running_mean[out_mask]
        if layer.running_var is not None:
            layer.running_var.data = layer.running_var[out_mask]

def _update_metadata_shape(metadata: TensorMetadata, new_num_channels: int) -> None:
    metadata.shape = (metadata.shape[0], new_num_channels, *metadata.shape[2:])

class StaticFeatMapPropagation(fx.Interpreter):
    def run_node(self, n : fx.Node) -> Any:
        output = super().run_node(n)

        n.output = output

        return output

class ChannelSlimmer:
    """
    Automatically slims the network, propagating pruned channels and removing layers as needed.
    Needs to follow the pruning group structure of ChannelStructure:
        - Pruning only done for common mask across group (for both input and output pruning)
        - Output pruning assumes the channel is always 0. This includes any subsequent BN layer!

    NOTE: Only tested on ResNet architectures.
    """
    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        model = deepcopy(model)
        model.eval()

        self.example_input = example_input[:1, ...]
        self.structure = ChannelStructure(model, self.example_input)

        self._upstream_dict = {
            k.target : v
            for k, v in self.structure._upstream_nodes_dict.items()
        }
        self._downstream_dict = {
            k.target : v
            for k, v in self.structure._downstream_nodes_dict.items()
        }

        assert not model.training, "Model needs to be in evaluation mode before slimming"

    @property
    def model(self) -> fx.GraphModule:
        return self.structure.gm

    @property
    def graph(self) -> fx.Graph:
        return self.structure.gm.graph

    def slim(self) -> nn.Module:
        # Add a temporary removal head as a substitute for dead layers
        with self.graph.inserting_before():
            removal_head = self.graph.placeholder("removal_head", torch.Tensor)

        # Slim input channels
        for group in self.structure.input_pruning_groups:
            self._slim_input_group(group, removal_head)

        # Slim output channels
        for group in self.structure.output_pruning_groups:
            self._slim_output_group(group, removal_head)

        # Remove unused layers (those upstream of fully pruned layers)
        self.graph.eliminate_dead_code()

        # Replace dead layers with static feature maps
        for dead_node in list(removal_head.users.keys()):
            self._replace_dead_layer(dead_node)

        # Delete the temporary removal head and recompile the model
        self.graph.erase_node(removal_head)
        self.model.recompile()

        # Propagate static feature maps
        self._propagate_static_featuremap()

        # Remove unnecessary modules
        self.model.delete_all_unused_submodules()

        return self.model

    def _slim_input_group(self, group: PruningGroup, removal_head: fx.Node) -> None:
        # Recover the group input mask
        group_input_mask = self._get_group_input_mask(group)

        if group_input_mask.all():
            return

        memo: Set[str] = set()
        num_in_channels = group_input_mask.sum().item()

        # Slim each member of the group with the group mask
        # Slim all of the upstream layers using those channels as well
        for layer_name in group:
            node: fx.Node = next(n for n in self.graph.nodes if n.target == layer_name)

            _slim_layer_input(
                self.model.get_submodule(layer_name), group_input_mask
            )
            _update_metadata_shape(node.in_metadata[0], num_in_channels)

            if group_input_mask.any():
                for upstream_node in self._upstream_dict[layer_name]:
                    if upstream_node.target not in memo:
                        _slim_layer_output(
                            self.model.get_submodule(upstream_node.target),
                            group_input_mask
                        )
                        _update_metadata_shape(upstream_node.out_metadata[0], num_in_channels)

                        memo.add(upstream_node.target)
            else:
                node.args = (removal_head,)

    def _slim_output_group(self, group: PruningGroup, removal_head: fx.Node) -> None:
        # Recover the group output mask
        group_output_mask = self._get_group_output_mask(group)

        if group_output_mask.all():
            return

        memo: Set[str] = set()
        num_out_channels = group_output_mask.sum().item()

        # Slim each member of the group with the group mask
        # Slim all of the downstream layers using those channels as well
        for layer_name in group:
            node: Optional[fx.Node] = next(
                (n for n in self.graph.nodes if n.target == layer_name),
                None
            )

            # Node already set for deletion
            if removal_head in node.args:
                continue

            _slim_layer_output(
                self.model.get_submodule(layer_name), group_output_mask
            )
            _update_metadata_shape(node.out_metadata[0], num_out_channels)

            for downstream_node in self._downstream_dict[layer_name]:
                # Node already set for deletion
                if removal_head in downstream_node.args:
                    continue
                elif downstream_node.target in memo:
                    continue

                if group_output_mask.any():
                    _slim_layer_input(
                        self.model.get_submodule(downstream_node.target),
                        group_output_mask
                    )
                    _update_metadata_shape(downstream_node.in_metadata[0], num_out_channels)
                else:
                    downstream_node.args = (removal_head,)

                memo.add(downstream_node.target)

    def _replace_dead_layer(self, dead_node: fx.Node) -> None:
        assert dead_node.op == "call_module", "Only modules can be dead nodes"
        layer = self.model.get_submodule(dead_node.target)

        out_metadata = dead_node.out_metadata[0]
        if layer.bias is None:
            output_fm = torch.zeros(out_metadata.shape, dtype=out_metadata.dtype, device=out_metadata.device).to(memory_format=out_metadata.memory_format)
        else:
            output_fm = layer.bias.view(1, -1).expand(1, -1, *out_metadata.shape[2:])

        parent_name, _, name = dead_node.target.rpartition(".")
        parent = self.model.get_submodule(parent_name)

        parent.register_buffer(f"{name}_fm_tmp", output_fm)

        with self.graph.inserting_after(dead_node):
            fm_node = self.graph.get_attr(f"{parent_name}.{name}_fm_tmp", torch.Tensor)

        dead_node.replace_all_uses_with(fm_node)
        self.graph.erase_node(dead_node)

    def _propagate_static_featuremap(self) -> None:
        StaticFeatMapPropagation(self.model).run(self.example_input)

        # Find which nodes depend on the input (aka not solely on the static feature maps)
        input_dependent: Set[str] = set()
        node: fx.Node
        for node in self.graph.nodes:
            if node.op == "placeholder":
                input_dependent.add(node.name)
            elif len(node.all_input_nodes) > 0 and any(n.name in input_dependent for n in node.all_input_nodes):
                input_dependent.add(node.name)

        # Add a temporary removal head as a substitute for removed layers
        with self.graph.inserting_before():
            removal_head = self.graph.placeholder("removal_head", torch.Tensor)

        # Update nodes that do not depend on the input
        for node in self.graph.nodes:
            if node.name in input_dependent:
                continue

            is_any_user_input_dependent = any(n.name in input_dependent for n in node.users.keys())
            if is_any_user_input_dependent:
                # If some users do depend on input, add as feature map
                parent_name, _, name = node.target.rpartition(".")
                parent = self.model.get_submodule(parent_name)

                # Under special circumstances, feature map can be channel-constant
                fm = node.output
                fm_channel = fm.view(fm.shape[1], -1)
                if (fm_channel[:, :1] == fm_channel).all():
                    # Feature map is constant across channels, so store more compactly
                    new_shape = (1, fm.shape[1], *([1] * (fm.ndim - 2)))
                    fm = fm_channel[:, 0].view(*new_shape)

                parent.register_buffer(f"{name}_fm", fm)

                with self.graph.inserting_before(node):
                    bias_node = self.graph.get_attr(f"{parent_name}.{name}_fm", torch.Tensor)

                node.replace_all_uses_with(bias_node)
                self.graph.erase_node(node)
            else:
                # If users don't depend on input (or don't exist), remove the node
                if node.name.endswith("fm_tmp"):
                    # Clean-up temporary feature map attribute
                    parent_name, _, name = node.target.rpartition(".")
                    parent = self.model.get_submodule(parent_name)
                    delattr(parent, name)

                # If all users also don't depend on input, delete node
                node.replace_all_uses_with(removal_head)
                self.graph.erase_node(node)

        self.graph.erase_node(removal_head)

        self.graph.eliminate_dead_code()
        self.model.recompile()

    def _get_group_input_mask(self, group: PruningGroup) -> torch.Tensor:
        # Recover the group input mask
        input_masks = []
        for layer_name in group:
            layer = self.model.get_submodule(layer_name)

            input_mask = _get_input_mask(layer)
            if input_mask is not None:
                input_masks.append(input_mask)

        group_input_mask = torch.stack(input_masks, dim=1).any(dim=1)

        return group_input_mask

    def _get_group_output_mask(self, group: PruningGroup) -> torch.Tensor:
        # Recover the group output mask
        output_masks = []
        for layer_name in group:
            layer = self.model.get_submodule(layer_name)

            output_mask = _get_output_mask(layer)
            if output_mask is not None:
                output_masks.append(output_mask)

        group_output_mask = torch.stack(output_masks, dim=1).any(dim=1)

        return group_output_mask
