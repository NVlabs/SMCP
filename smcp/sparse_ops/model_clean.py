# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from torch import nn

from smcp.sparse_ops.channel_slimmer import ChannelSlimmer

def clean_model(
    model: nn.Module, example_input: torch.Tensor,
    should_slim: bool, should_fuse: bool = False, convert_torchscript: bool = False
) -> nn.Module:
    if should_slim:
        model = ChannelSlimmer(model, example_input).slim()
    if convert_torchscript:
        model = torch.jit.script(model)

    return model
