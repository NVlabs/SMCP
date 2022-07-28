# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from abc import ABC, abstractmethod
from enum import Enum
import math
import pickle as pkl
from typing import Dict, Generator, Optional, Set
import warnings

import torch
from torch import nn

class CostingType(Enum):
    Param = "Param"
    Flop = "Flop"
    Latency = "Latency"

class Coster(ABC):
    @abstractmethod
    def get_cost(self, layer: nn.Module, input_shape: torch.Size, in_channels: Optional[int] = None, out_channels: Optional[int] = None) -> float:
        pass

    def get_cumulative_input_channel_cost(self, layer: nn.Module, input_shape: torch.Size, eff_out_mask: torch.Tensor) -> torch.Tensor:
        eff_out_channels = eff_out_mask.sum().item()
        device = eff_out_mask.device

        if isinstance(layer, nn.Conv2d):
            in_channels = layer.in_channels
        elif isinstance(layer, nn.Linear):
            in_channels = layer.in_features
        else:
            raise NotImplementedError(f"Input channel cost not implemented for layer of type {type(layer)}")

        return torch.tensor(
            [self.get_cost(layer, input_shape, c, eff_out_channels) for c in range(1, in_channels + 1)],
            device=device, dtype=torch.float32
        )

    def get_cumulative_output_channel_cost(self, layer: nn.Module, input_shape: torch.Size, eff_in_mask: torch.Tensor) -> torch.Tensor:
        eff_in_channels = eff_in_mask.sum().item()
        device = eff_in_mask.device

        if isinstance(layer, nn.Conv2d):
            out_channels = layer.out_channels
        elif isinstance(layer, nn.Linear):
            out_channels = layer.out_features
        else:
            raise NotImplementedError(f"Output channel cost not implemented for layer of type {type(layer)}")

        return torch.tensor(
            [self.get_cost(layer, input_shape, eff_in_channels, c) for c in range(1, out_channels + 1)],
            device=device, dtype=torch.float32
        )

class ParamCoster(Coster):
    def get_cost(self, layer: nn.Module, _: torch.Size, in_channels: Optional[int] = None, out_channels: Optional[int] = None) -> float:
        cost = 0
        if isinstance(layer, nn.Conv2d):
            assert layer.groups == 1

            out_channels = layer.out_channels if out_channels is None else out_channels
            in_channels = layer.in_channels if in_channels is None else in_channels

            cost = (out_channels // layer.groups) * in_channels * math.prod(layer.kernel_size)
        elif isinstance(layer, nn.Linear):
            out_channels = layer.out_features if out_channels is None else out_channels
            in_channels = layer.in_features if in_channels is None else in_channels

            cost = out_channels * in_channels
        else:
            raise NotImplementedError

        return cost / 1e6  # Returns in unit of millions

class FlopCoster(ParamCoster):
    def get_cost(self, layer: nn.Module, input_shape: torch.Size, in_channels: Optional[int] = None, out_channels: Optional[int] = None) -> float:
        num_params = super().get_cost(layer, input_shape, in_channels, out_channels)

        cost = 0
        if isinstance(layer, nn.Conv2d):
            assert layer.groups == 1

            out_size = torch.div(
                torch.tensor(input_shape[1:]) + 2 * torch.tensor(layer.padding) \
                - torch.tensor(layer.dilation) * (torch.tensor(layer.kernel_size) - 1) \
                + torch.tensor(layer.stride) - 1,
                torch.tensor(layer.stride),
                rounding_mode="trunc"
            ).int()
            num_patches = out_size[0] * out_size[1]

            cost = num_patches * num_params
        elif isinstance(layer, nn.Linear):
            cost = num_params
        else:
            raise NotImplementedError

        return cost / 1e3  # Returns in unit of Gigaflops

class LatencyCoster(Coster):
    # Key is in one of several formats:
    #  1. <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>
    #  2. <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>_<groups>
    # Value is the corresponding latency
    lookup_table: Dict[str, float]

    def __init__(self, latency_file: str, ignore_lut_warnings: bool = False):
        with open(latency_file, "rb") as f:
            self.lookup_table = pkl.load(f)

        uniq_batch_sizes = set(int(key.split("_")[0]) for key in self.lookup_table.keys())
        assert len(uniq_batch_sizes) == 1, "Lookup table should only have a single batch size"

        self.inference_batch_size = list(uniq_batch_sizes)[0]

        # Infer the key format
        example_key = next(iter(self.lookup_table.keys()))
        key_size = len(example_key.split("_"))
        if key_size == 6:
            self.key_format = "no_groups"
        elif key_size == 7:
            self.key_format = "groups"
        else:
            raise NotImplementedError

        self.has_warned: Set[str] = set()
        self.ignore_lut_warnings = ignore_lut_warnings

    def get_cost(self, layer: nn.Module, input_shape: torch.Size, in_channels: Optional[int] = None, out_channels: Optional[int] = None) -> float:
        if isinstance(layer, nn.Conv2d):
            fmap_size = input_shape[1]

            out_channels = layer.out_channels if out_channels is None else out_channels
            in_channels = layer.in_channels if in_channels is None else in_channels

            assert layer.groups == 1 or (layer.groups == layer.in_channels), "Latency cost only implemented for normal and depthwise convolutions"
            is_depthwise = (layer.groups == layer.in_channels)

            kernel = layer.kernel_size[0]
            assert all(kernel == k for k in layer.kernel_size), "Only support latency lookup for square kernels"

            stride = layer.stride[0]
            assert all(stride == s for s in layer.stride), "Only support latency lookup for equal strides"

            return self._get_latency(fmap_size, in_channels, out_channels, kernel, stride, is_depthwise)
        elif isinstance(layer, nn.Linear):
            fmap_size = input_shape[0]

            out_channels = layer.out_features if out_channels is None else out_channels
            in_channels = layer.in_features if in_channels is None else in_channels

            kernel = 1
            stride = 1

            return self._get_latency(fmap_size, in_channels, out_channels, kernel, stride, False)

        return 0

    def _get_latency(
        self, input_size: int, in_channels: int, out_channels: int, kernel: int, stride: int, is_depthwise: bool
    ) -> float:
        """
        Get the latency from the lookup table

        Args:
            input_size: the input feature map size
            in_channels: number of input channels
            out_channels: number of output channels
            kernel: kernel size
            stride: stride
            is_depthwise: whether the operation is a depthwise convolution

        Returns:
            lat: the corresponding latency
        """

        if in_channels <= 0 or out_channels <= 0:
            return 0

        for possible_key in self._get_possible_keys(input_size, in_channels, out_channels, kernel, stride, is_depthwise):
            if possible_key in self.lookup_table:
                return self.lookup_table[possible_key]

        orig_key = self._get_key(input_size, in_channels, out_channels, kernel, stride, in_channels if is_depthwise else 1)
        if not self.ignore_lut_warnings and (orig_key not in self.has_warned):
            warnings.warn(f"Cannot find key (or any nearby key) in the lookup table for {orig_key}. Assuming 0 latency.")
            self.has_warned.add(orig_key)

        return 0

    def _get_possible_keys(
        self, input_size: int, in_channels: int, out_channels: int, kernel: int, stride: int, is_depthwise: bool
    ) -> Generator[str, None, None]:
        groups = in_channels if is_depthwise else 1

        yield self._get_key(input_size, in_channels, out_channels, kernel, stride, groups)

        # Assume equally expensive until next multiple of 8
        alt_in_channels = in_channels + (8 - in_channels % 8)
        alt_out_channels = out_channels + (8 - out_channels % 8)
        alt_groups = alt_in_channels if is_depthwise else 1

        if in_channels % 8 != 0:
            yield self._get_key(input_size, alt_in_channels, out_channels, kernel, stride, alt_groups)

        if out_channels % 8 != 0:
            yield self._get_key(input_size, in_channels, alt_out_channels, kernel, stride, groups)

        if (in_channels % 8 != 0) and (out_channels % 8 != 0):
            yield self._get_key(input_size, alt_in_channels, alt_out_channels, kernel, stride, alt_groups)

    def _get_key(self, input_size: int, in_channels: int, out_channels: int, kernel: int, stride: int, groups: int) -> str:
        if self.key_format == "no_groups":
            assert groups == 1
            return f"{self.inference_batch_size}_{in_channels}_{out_channels}_{input_size}_{kernel}_{stride}"
        elif self.key_format == "groups":
            return f"{self.inference_batch_size}_{in_channels}_{out_channels}_{input_size}_{kernel}_{stride}_{groups}"
        else:
            raise NotImplementedError

def create_coster(
    costing_type: CostingType, latency_table: Optional[str] = None, ignore_lut_warnings: bool = False
) -> Coster:
    if costing_type == CostingType.Param:
        return ParamCoster()
    elif costing_type == CostingType.Flop:
        return FlopCoster()
    elif costing_type == CostingType.Latency:
        assert latency_table is not None, "Latency coster requires a latency table"

        return LatencyCoster(latency_table, ignore_lut_warnings)

    raise NotImplementedError(f"Costing type {costing_type} invalid")
