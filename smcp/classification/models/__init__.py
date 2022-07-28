# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Union

from torch import nn

from smcp.classification.models.mobilenetv1 import get_mobilenetv1_model
from smcp.classification.models.resnet import get_resnet_model

def get_classification_model(
    arch: str, num_classes: int, pretrained: Union[bool, str] = False
) -> nn.Module:
    if "resnet" in arch:
        return get_resnet_model(arch, num_classes, pretrained, progress=False)
    elif arch == "mobilenetV1":
        return get_mobilenetv1_model(num_classes, pretrained)

    raise NotImplementedError
