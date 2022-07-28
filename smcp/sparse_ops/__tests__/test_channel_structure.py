# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Callable, Dict, List, Tuple

import pytest
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50

from smcp.sparse_ops.channel_structure import ChannelStructure

ModelFn = Callable[[], nn.Module]

def test_resnet50_input_shapes():
    model = resnet50()

    fake_input = torch.randn((1, 3, 224, 224))
    structure = ChannelStructure(model, fake_input)

    input_metadata_dict = structure.input_metadata_dict

    expected_shape_dict = {
        "conv1": torch.Size([1, 3, 224, 224]),

        "layer1.0.conv1": torch.Size([1, 64, 56, 56]),
        "layer1.0.conv2": torch.Size([1, 64, 56, 56]),
        "layer1.0.conv3": torch.Size([1, 64, 56, 56]),
        "layer1.1.conv1": torch.Size([1, 256, 56, 56]),
        "layer1.1.conv2": torch.Size([1, 64, 56, 56]),
        "layer1.1.conv3": torch.Size([1, 64, 56, 56]),
        "layer1.2.conv1": torch.Size([1, 256, 56, 56]),
        "layer1.2.conv2": torch.Size([1, 64, 56, 56]),
        "layer1.2.conv3": torch.Size([1, 64, 56, 56]),
        "layer1.0.downsample.0": torch.Size([1, 64, 56, 56]),

        "layer2.0.conv1": torch.Size([1, 256, 56, 56]),
        "layer2.0.conv2": torch.Size([1, 128, 56, 56]),
        "layer2.0.conv3": torch.Size([1, 128, 28, 28]),
        "layer2.1.conv1": torch.Size([1, 512, 28, 28]),
        "layer2.1.conv2": torch.Size([1, 128, 28, 28]),
        "layer2.1.conv3": torch.Size([1, 128, 28, 28]),
        "layer2.2.conv1": torch.Size([1, 512, 28, 28]),
        "layer2.2.conv2": torch.Size([1, 128, 28, 28]),
        "layer2.2.conv3": torch.Size([1, 128, 28, 28]),
        "layer2.3.conv1": torch.Size([1, 512, 28, 28]),
        "layer2.3.conv2": torch.Size([1, 128, 28, 28]),
        "layer2.3.conv3": torch.Size([1, 128, 28, 28]),
        "layer2.0.downsample.0": torch.Size([1, 256, 56, 56]),

        "layer3.0.conv1": torch.Size([1, 512, 28, 28]),
        "layer3.0.conv2": torch.Size([1, 256, 28, 28]),
        "layer3.0.conv3": torch.Size([1, 256, 14, 14]),
        "layer3.1.conv1": torch.Size([1, 1024, 14, 14]),
        "layer3.1.conv2": torch.Size([1, 256, 14, 14]),
        "layer3.1.conv3": torch.Size([1, 256, 14, 14]),
        "layer3.2.conv1": torch.Size([1, 1024, 14, 14]),
        "layer3.2.conv2": torch.Size([1, 256, 14, 14]),
        "layer3.2.conv3": torch.Size([1, 256, 14, 14]),
        "layer3.3.conv1": torch.Size([1, 1024, 14, 14]),
        "layer3.3.conv2": torch.Size([1, 256, 14, 14]),
        "layer3.3.conv3": torch.Size([1, 256, 14, 14]),
        "layer3.4.conv1": torch.Size([1, 1024, 14, 14]),
        "layer3.4.conv2": torch.Size([1, 256, 14, 14]),
        "layer3.4.conv3": torch.Size([1, 256, 14, 14]),
        "layer3.5.conv1": torch.Size([1, 1024, 14, 14]),
        "layer3.5.conv2": torch.Size([1, 256, 14, 14]),
        "layer3.5.conv3": torch.Size([1, 256, 14, 14]),
        "layer3.0.downsample.0": torch.Size([1, 512, 28, 28]),

        "layer4.0.conv1": torch.Size([1, 1024, 14, 14]),
        "layer4.0.conv2": torch.Size([1, 512, 14, 14]),
        "layer4.0.conv3": torch.Size([1, 512, 7, 7]),
        "layer4.1.conv1": torch.Size([1, 2048, 7, 7]),
        "layer4.1.conv2": torch.Size([1, 512, 7, 7]),
        "layer4.1.conv3": torch.Size([1, 512, 7, 7]),
        "layer4.2.conv1": torch.Size([1, 2048, 7, 7]),
        "layer4.2.conv2": torch.Size([1, 512, 7, 7]),
        "layer4.2.conv3": torch.Size([1, 512, 7, 7]),
        "layer4.0.downsample.0": torch.Size([1, 1024, 14, 14]),

        "fc": torch.Size([1, 2048])
    }

    assert set(input_metadata_dict.keys()) == set(expected_shape_dict.keys()), "Input shape dict has different layer keys"

    for layer, metadata in input_metadata_dict.items():
        shape = metadata.shape
        assert shape == expected_shape_dict[layer], f"Layer {layer}'s shape {shape} does not match the expected shape {expected_shape_dict[layer]}"

@pytest.mark.parametrize("model_fn", [resnet18, resnet50])
def test_resnet_batchnorm_dict(model_fn: ModelFn):
    model = model_fn()
    structure = ChannelStructure(model)

    batchnorm_dict = structure.associated_batchnorm_dict

    expected_batchnorm_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            bn_name = name.replace("conv", "bn") if "conv" in name else name.replace("downsample.0", "downsample.1")
            expected_batchnorm_dict[name] = bn_name

    assert set(batchnorm_dict.keys()) == set(expected_batchnorm_dict.keys()), "Batchnorm dict has different layer keys"

    for layer_name, bn_name in batchnorm_dict.items():
        assert bn_name == expected_batchnorm_dict[layer_name], f"Layer {layer_name}'s batchnorm layer {bn_name} does not match the expected shape {expected_batchnorm_dict[layer_name]}"

def _test_groups_match_nodesdict(groups: List[Tuple[str, ...]], nodes_dict: Dict[str, List[str]]) -> None:
    # Build group id lookup dict
    groupid_dict = {}
    for idx, group in enumerate(groups):
        for layer_name in group:
            groupid_dict[layer_name] = idx

    for node, cnctd_nodes in nodes_dict.items():
        ids = [
            groupid_dict[cnctd_node]
            for cnctd_node in cnctd_nodes
        ]

        assert len(set(ids)) <= 1, f"Connected nodes of {node} are not in the same group"

@pytest.mark.parametrize("model_fn", [resnet18, resnet50])
def test_resnet_input_groups(model_fn: ModelFn):
    model = model_fn()
    structure = ChannelStructure(model)

    input_pruning_groups = structure.input_pruning_groups
    downstream_dict = structure.downstream_producers_dict

    # Need to assert every layer"s outputs fall in the same group for input pruning
    _test_groups_match_nodesdict(input_pruning_groups, downstream_dict)

@pytest.mark.parametrize("model_fn", [resnet18, resnet50])
def test_resnet_output_groups(model_fn: ModelFn):
    model = model_fn()
    structure = ChannelStructure(model)

    output_pruning_groups = structure.output_pruning_groups
    upstream_dict = structure.upstream_producers_dict

    # Need to assert every layer's inputs fall in the same group for output pruning
    _test_groups_match_nodesdict(output_pruning_groups, upstream_dict)

def _assert_groups_match(groups: List[Tuple[str, ...]], expected: List[str]) -> bool:
    value = [
        ", ".join(sorted(g)) for g in groups
    ]
    expected = [
        ", ".join(sorted(s.split(", "))) for s in expected
    ]

    diff = set(value) ^ set(expected)
    return not diff

def test_resnet50_input_pruning():
    model = resnet50()
    structure = ChannelStructure(model)

    input_pruning_groups = structure.input_pruning_groups

    expected = [
        "conv1",
        "layer1.0.conv1, layer1.0.downsample.0",
        "layer1.0.conv2",
        "layer1.0.conv3",
        "layer1.1.conv1, layer1.2.conv1, layer2.0.conv1, layer2.0.downsample.0",
        "layer1.1.conv2",
        "layer1.1.conv3",
        "layer1.2.conv2",
        "layer1.2.conv3",
        "layer2.0.conv2",
        "layer2.0.conv3",
        "layer2.1.conv1, layer2.2.conv1, layer2.3.conv1, layer3.0.conv1, layer3.0.downsample.0",
        "layer2.1.conv2",
        "layer2.1.conv3",
        "layer2.2.conv2",
        "layer2.2.conv3",
        "layer2.3.conv2",
        "layer2.3.conv3",
        "layer3.0.conv2",
        "layer3.0.conv3",
        "layer3.1.conv1, layer3.2.conv1, layer3.3.conv1, layer3.4.conv1, layer3.5.conv1, layer4.0.conv1, layer4.0.downsample.0",
        "layer3.1.conv2",
        "layer3.1.conv3",
        "layer3.2.conv2",
        "layer3.2.conv3",
        "layer3.3.conv2",
        "layer3.3.conv3",
        "layer3.4.conv2",
        "layer3.4.conv3",
        "layer3.5.conv2",
        "layer3.5.conv3",
        "layer4.0.conv2",
        "layer4.0.conv3",
        "layer4.1.conv1, layer4.2.conv1, fc",
        "layer4.1.conv2",
        "layer4.1.conv3",
        "layer4.2.conv2",
        "layer4.2.conv3",
    ]

    assert _assert_groups_match(input_pruning_groups, expected), "Resnet50 input pruning grouping incorrectly discovered"

def test_resnet50_output_pruning():
    model = resnet50()
    structure = ChannelStructure(model)

    output_pruning_groups = structure.output_pruning_groups

    expected = [
        "conv1",
        "layer1.0.conv1",
        "layer1.0.conv2",
        "layer1.1.conv1",
        "layer1.1.conv2",
        "layer1.2.conv1",
        "layer1.2.conv2",
        "layer1.0.downsample.0, layer1.0.conv3, layer1.1.conv3, layer1.2.conv3",
        "layer2.0.conv1",
        "layer2.0.conv2",
        "layer2.1.conv1",
        "layer2.1.conv2",
        "layer2.2.conv1",
        "layer2.2.conv2",
        "layer2.3.conv1",
        "layer2.3.conv2",
        "layer2.0.downsample.0, layer2.0.conv3, layer2.1.conv3, layer2.2.conv3, layer2.3.conv3",
        "layer3.0.conv1",
        "layer3.0.conv2",
        "layer3.1.conv1",
        "layer3.1.conv2",
        "layer3.2.conv1",
        "layer3.2.conv2",
        "layer3.3.conv1",
        "layer3.3.conv2",
        "layer3.4.conv1",
        "layer3.4.conv2",
        "layer3.5.conv1",
        "layer3.5.conv2",
        "layer3.0.downsample.0, layer3.0.conv3, layer3.1.conv3, layer3.2.conv3, layer3.3.conv3, layer3.4.conv3, layer3.5.conv3",
        "layer4.0.conv1",
        "layer4.0.conv2",
        "layer4.1.conv1",
        "layer4.1.conv2",
        "layer4.2.conv1",
        "layer4.2.conv2",
        "layer4.0.downsample.0, layer4.0.conv3, layer4.1.conv3, layer4.2.conv3",
        "fc"
    ]

    assert _assert_groups_match(output_pruning_groups, expected), "Resnet50 output pruning grouping incorrectly discovered"

def test_resnet50_upstream_producers():
    model = resnet50()
    structure = ChannelStructure(model)

    upstream_producers = structure.upstream_producers_dict

    partial_expected = {
        "layer1.2.conv1": ["layer1.0.downsample.0", "layer1.0.conv3", "layer1.1.conv3"],
        "layer2.1.bn2": ["layer2.1.conv2"],
        "layer3.1.conv1": ["layer3.0.downsample.0", "layer3.0.conv3"],
        "layer4.2.conv2": ["layer4.2.conv1"]
    }

    for name, upstream_names in partial_expected.items():
        assert name in upstream_producers, f"{name} not producers dict"
        assert set(upstream_producers[name]) == set(upstream_names), f"Upstream producers don't agree for {name}"

def test_mobilenetv1_input_groups():
    model = resnet50()
    structure = ChannelStructure(model)

    input_pruning_groups = structure.input_pruning_groups

    expected = [
        ("features.conv_bn.conv",),
        ("features.conv_dw1.conv2",),
        ("features.conv_dw2.conv2",),
        ("features.conv_dw3.conv2",),
        ("features.conv_dw4.conv2",),
        ("features.conv_dw5.conv2",),
        ("features.conv_dw6.conv2",),
        ("features.conv_dw7.conv2",),
        ("features.conv_dw8.conv2",),
        ("features.conv_dw9.conv2",),
        ("features.conv_dw10.conv2",),
        ("features.conv_dw11.conv2",),
        ("features.conv_dw12.conv2",),
        ("features.conv_dw13.conv2",)
    ]

    assert _assert_groups_match(input_pruning_groups, expected), "MobileNetV1 input pruning grouping incorrectly discovered"
