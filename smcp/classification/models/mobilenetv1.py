# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from collections import OrderedDict
from typing import Union

import torch
from torch import nn

class MoblieNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MoblieNetV1, self).__init__()
        self.num_classes = num_classes

        def conv_bn(inp: int, oup: int, stride : int) -> nn.Sequential:
            layers = [('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
                      ('bn', nn.BatchNorm2d(oup)),
                      ('relu', nn.ReLU(inplace=True))]
            return nn.Sequential(OrderedDict(layers))

        def conv_dw(inp: int, oup: int, stride: int) -> nn.Sequential:
            layers = [('conv1', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
                      ('bn1', nn.BatchNorm2d(inp)),
                      ('relu1', nn.ReLU(inplace=True)),
                      ('conv2', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                      ('bn2', nn.BatchNorm2d(oup)),
                      ('relu2', nn.ReLU(inplace=True))]
            return nn.Sequential(OrderedDict(layers))

        layers = [('conv_bn', conv_bn(3, 32, 2)),
                  ('conv_dw1', conv_dw(32, 64, 1)),
                  ('conv_dw2', conv_dw(64, 128, 2)),
                  ('conv_dw3', conv_dw(128, 128, 1)),
                  ('conv_dw4', conv_dw(128, 256, 2)),
                  ('conv_dw5', conv_dw(256, 256, 1)),
                  ('conv_dw6', conv_dw(256, 512, 2)),
                  ('conv_dw7', conv_dw(512, 512, 1)),
                  ('conv_dw8', conv_dw(512, 512, 1)),
                  ('conv_dw9', conv_dw(512, 512, 1)),
                  ('conv_dw10', conv_dw(512, 512, 1)),
                  ('conv_dw11', conv_dw(512, 512, 1)),
                  ('conv_dw12', conv_dw(512, 1024, 2)),
                  ('conv_dw13', conv_dw(1024, 1024, 1)),
                  ('avg_pool', nn.AvgPool2d(7))]
        self.features = nn.Sequential(OrderedDict(layers))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_mobilenetv1_model(num_classes: int, pretrained: Union[bool, str] = False) -> nn.Module:
    if isinstance(pretrained, str):
        state_dict = torch.load(pretrained)
        pretrained_num_classes = state_dict["fc.weight"].shape[0]

        model = MoblieNetV1(pretrained_num_classes)
        model.load_state_dict(state_dict)

        if num_classes != pretrained_num_classes:
            # Pretrained model not for this dataset
            # Reuse everything but the final FC layer
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    return MoblieNetV1(num_classes)
