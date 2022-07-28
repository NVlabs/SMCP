# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import List

import pytest
import torch
from torchvision.models.resnet import ResNet, resnet50

from smcp.sparse_ops.channel_slimmer import ChannelSlimmer

def apply_input_mask(model: ResNet, name: str, ignore_idxs: List[int]) -> None:
    layer = model.get_submodule(name)
    layer.weight.data[:, ignore_idxs, ...] = 0

def apply_output_mask(model: ResNet, name: str, ignore_idxs: torch.Tensor) -> None:
    layer = model.get_submodule(name)
    layer.weight.data[ignore_idxs, ...] = 0

    if "conv" in name:
        bn_name = name.replace("conv", "bn")
    elif "downsample" in name:
        bn_name = f"{name[:-2]}.1"
    else:
        raise NotImplementedError

    bn_layer = model.get_submodule(bn_name)
    bn_layer.weight.data[ignore_idxs] = 0
    bn_layer.bias.data[ignore_idxs] = 0
    bn_layer.running_mean.data[ignore_idxs] = 0
    bn_layer.running_var.data[ignore_idxs] = 1

def test_resnet50_input_slimming_single():
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    apply_input_mask(model, "layer2.1.conv3", [4, 8, 31, 36])

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    assert slim_model.get_submodule("layer2.1.conv3").in_channels == 124, "layer2.1.conv3 not slimmed"
    assert slim_model.get_submodule("layer2.1.bn2").num_features == 124, "layer2.1.bn2 not slimmed"
    assert slim_model.get_submodule("layer2.1.conv2").out_channels == 124, "layer2.1.conv2 not slimmed"

def test_resnet50_output_slimming_single():
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    apply_output_mask(model, "layer1.1.conv1", [0,1,5,9])

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    assert slim_model.get_submodule("layer1.1.conv1").out_channels == 60, "layer1.1.conv1 not slimmed"
    assert slim_model.get_submodule("layer1.1.bn1").num_features == 60, "layer1.1.bn1 not slimmed"
    assert slim_model.get_submodule("layer1.1.conv2").in_channels == 60, "layer1.1.conv2 not slimmed"

def test_resnet50_input_slimming_group():
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    group = ["layer1.1.conv1", "layer1.2.conv1", "layer2.0.conv1", "layer2.0.downsample.0"]
    for idx, layer_name in enumerate(group):
        # Test adding extra zero channels that aren't common to the whole group
        zeroed_idxs = [0,1,5,9,10+idx,31+idx]
        apply_input_mask(model, layer_name, zeroed_idxs)

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    for layer_name in group:
        assert slim_model.get_submodule(layer_name).in_channels == 252, f"{layer_name} not slimmed"

def test_resnet50_output_slimming_group():
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    group = ["layer1.0.downsample.0", "layer1.0.conv3", "layer1.1.conv3", "layer1.2.conv3"]
    for idx, layer_name in enumerate(group):
        # Test adding extra zero channels that aren't common to the whole group
        zeroed_idxs = [0,1,5,9,10+idx,31+idx]

        apply_output_mask(model, layer_name, zeroed_idxs)

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    for layer_name in group:
        assert slim_model.get_submodule(layer_name).out_channels == 252, f"{layer_name} not slimmed"

@pytest.mark.parametrize(["name", "num_in_channels"], [("conv2", 64), ("conv3", 64)])
def test_resnet50_input_layer_removal(name: str, num_in_channels: int):
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    apply_input_mask(model, f"layer1.1.{name}", list(range(num_in_channels)))

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    with pytest.raises(AttributeError, match=r"Module has no attribute `conv3`"):
        slim_model.get_submodule("layer1.1.conv3")

    assert isinstance(slim_model.get_buffer("layer1.1.bn3_fm"), torch.Tensor), "Branch replaced by a feature map"

@pytest.mark.parametrize(["name", "num_out_channels"], [("conv1", 64), ("conv2", 64)])
def test_resnet50_output_layer_removal(name: str, num_out_channels: int):
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    apply_output_mask(model, f"layer1.1.{name}", list(range(num_out_channels)))

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    with pytest.raises(AttributeError, match=r"Module has no attribute `conv3`"):
        slim_model.get_submodule("layer1.1.conv3")

    assert isinstance(slim_model.get_buffer("layer1.1.bn3_fm"), torch.Tensor), "Branch replaced by a feature map"

def test_resnet50_complex_layer_removal():
    input = torch.randn(32, 3, 224, 224)
    model = resnet50(pretrained=True)
    model.eval()

    apply_output_mask(model, "layer1.1.conv2", list(range(64)))
    apply_input_mask(model, "layer1.1.conv3", list(range(64)))

    apply_output_mask(model, "layer2.0.conv2", list(range(128)))

    apply_output_mask(model, "layer3.4.conv2", [0, 24, 124, 238])
    apply_input_mask(model, "layer3.4.conv3", [12, 24])

    apply_input_mask(model, "layer4.2.conv2", [431, 510])

    slim_model = ChannelSlimmer(model, input[:1, ...]).slim()

    output = model(input)
    slim_output = slim_model(input)

    assert slim_output.shape == (32, 1000), "Network outputs wrong shape"
    assert torch.isclose(output, slim_output, atol=1e-5).all(), "Slimmed network gives different output"

    with pytest.raises(AttributeError, match=r"Module has no attribute `conv1`"):
        slim_model.get_submodule("layer1.1.conv1")
    with pytest.raises(AttributeError, match=r"Module has no attribute `conv1`"):
        slim_model.get_submodule("layer2.0.conv1")

    assert slim_model.get_submodule("layer3.4.conv2").out_channels == 251, "Wrong number of out channels"
    assert slim_model.get_submodule("layer3.4.conv3").in_channels == 251, "Wrong number of in channels"

    assert slim_model.get_submodule("layer4.2.conv1").out_channels == 510, "Wrong number of out channels"
    assert slim_model.get_submodule("layer4.2.conv2").in_channels == 510, "Wrong number of in channels"
