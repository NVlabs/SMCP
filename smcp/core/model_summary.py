# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from functools import partial
import math
from typing import Callable, Dict, Tuple

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

@torch.no_grad()
def time_inference(
    model: nn.Module, dataloader: DataLoader, num_batches: int = 30, warmup: int = 10
) -> float:
    cudnn.benchmark = True
    cudnn.deterministic = True

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    device = torch.device("cuda")
    model = model.to(device)

    model.eval()

    times = []
    for i, (input, target) in enumerate(dataloader):
        input = input.to(device)

        start_evt.record()
        output = model(input)
        end_evt.record()

        torch.cuda.synchronize()
        elapsed_time = start_evt.elapsed_time(end_evt)

        if i < warmup:
            continue

        times.append(elapsed_time)

        if i >= warmup + num_batches:
            break

    return sum(times) / len(times)

def _count_params(model: nn.Module) -> int:
    num_params = 0
    for param in model.parameters(recurse=True):
        num_params += param.numel()

    return num_params

def _get_conv2d_macs(layer: nn.Conv2d, input: torch.Tensor, output: torch.Tensor) -> int:
    # Multiply-accumulate operations: MACs = volume(OFM) * (#IFM * K^2) / #Groups
    # Bias is ignored
    return math.prod(output.shape) * (
        layer.in_channels / layer.groups * layer.kernel_size[0] * layer.kernel_size[1]
    )

def _get_linear_macs(layer: nn.Linear, input: torch.Tensor, output: torch.Tensor) -> int:
    # Multiply-accumulate operations: MACs = #IFM * #OFM
    # Bias is ignored
    return layer.in_features * layer.out_features

def _count_macs(model: nn.Module, input: torch.Tensor) -> int:
    print("Only counting MACs from Conv2d and linear layers")

    macs_dict: Dict[str, int] = {}
    def mac_hook(name: str, mac_counter: Callable[..., int], *args, **kwargs):
        macs_dict[name] = mac_counter(*args, **kwargs)

    hook_handles = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            hook_handles.append(
                layer.register_forward_hook(partial(mac_hook, name, _get_conv2d_macs))
            )
        elif isinstance(layer, torch.nn.Linear):
            hook_handles.append(
                layer.register_forward_hook(partial(mac_hook, name, _get_linear_macs))
            )

    # Now run the forward path (on single batch) and collect the data
    model.eval()
    model(input[:1, ...])

    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    return sum(macs_dict.values())

# Taken from from https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/utils.py
def clever_format(num: int, format: str = "%.2f") -> str:
    if num > 1e12:
        return (format % (num / 1e12)) + "T"
    elif num > 1e9:
        return (format % (num / 1e9)) + "G"
    elif num > 1e6:
        return (format % (num / 1e6)) + "M"
    elif num > 1e3:
        return (format % (num / 1e3)) + "K"

    return format % num + "B"

@torch.no_grad()
def model_summary(model: nn.Module, dataloader: DataLoader) -> Tuple[str, str]:
    input, _ = next(iter(dataloader))

    device = torch.device("cuda")
    input = input.to(device)
    model.to(device)

    # Count params
    params = _count_params(model)

    # Count macs
    macs = _count_macs(model, input)

    return clever_format(macs), clever_format(params)
