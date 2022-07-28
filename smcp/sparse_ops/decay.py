# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Union

import torch
from torch import nn

from smcp.sparse_ops.parameter_masking import has_mask, get_mask, get_orig

@torch.jit.script
def _apply_decay(param: nn.Parameter, mask: torch.Tensor, pruned_decay: float) -> None:
    param.grad.add_(~mask * param, alpha=pruned_decay)

def _apply_weight_bias_pruned_decay(layer: Union[nn.Conv2d, nn.Linear], pruned_decay: float) -> None:
    if has_mask(layer, "weight"):
        _apply_decay(get_orig(layer, "weight"), get_mask(layer, "weight"), pruned_decay)
    if has_mask(layer, "bias"):
        _apply_decay(get_orig(layer, "bias"), get_mask(layer, "bias"), pruned_decay)

def apply_pruned_decay(module: nn.Module, pruned_decay: float) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        _apply_weight_bias_pruned_decay(module, pruned_decay)
