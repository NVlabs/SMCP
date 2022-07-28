# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# Adapted from https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

from typing import Union

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, model_urls

def get_resnet_model(arch: str, num_classes: int, pretrained: Union[bool, str] = False, progress: bool = True, **kwargs) -> ResNet:
    # Set building block
    if arch in ["resnet18", "resnet34"]:
        block = BasicBlock
    else:
        block = Bottleneck

    # Set number of layers
    if arch == "resnet18":
        layers = [2, 2, 2, 2]
    elif arch in ["resnet34", "resnet50", "resnext50_32x4d", "wide_resnet50_2"]:
        layers = [3, 4, 6, 3]
    elif arch in ["resnet101", "resnext101_32x8d", "wide_resnet101_2"]:
        layers = [3, 4, 23, 3]
    elif arch == "resnet152":
        layers = [3, 8, 36, 3]
    else:
        raise NotImplementedError

    # Set groups and widths
    if arch == "resnext50_32x4d":
        kwargs["groups"] = 32
        kwargs["width_per_group"] = 4
    elif arch == "resnext101_32x8d":
        kwargs["groups"] = 32
        kwargs["width_per_group"] = 8
    elif arch in ["wide_resnet50_2", "wide_resnet101_2"]:
        kwargs["width_per_group"] = 64 * 2

    IMAGENET_NUM_CLASSES = 1000

    if pretrained == True:
        model = ResNet(block, layers, num_classes=IMAGENET_NUM_CLASSES, zero_init_residual=True, **kwargs)
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)

        if num_classes != IMAGENET_NUM_CLASSES:
            # Pretrained model not directly available for other datasets
            # Reuse everything but the final FC layer
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model
    elif isinstance(pretrained, str):
        state_dict = torch.load(pretrained)
        pretrained_num_classes = state_dict["fc.weight"].shape[0]

        model = ResNet(block, layers, num_classes=pretrained_num_classes, zero_init_residual=True, **kwargs)
        model.load_state_dict(state_dict)

        if num_classes != pretrained_num_classes:
            # Pretrained model not for this dataset
            # Reuse everything but the final FC layer
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    return ResNet(block, layers, num_classes=num_classes, zero_init_residual=True, **kwargs)
