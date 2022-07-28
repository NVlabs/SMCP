# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# Adapted from by https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import fx

@dataclass
class TensorMetadata:
    shape : torch.Size
    dtype : torch.dtype
    device: torch.device
    memory_format : Optional[torch.memory_format]

    @staticmethod
    def extract_metadata(result : torch.Tensor) -> TensorMetadata:
        shape = result.shape
        dtype = result.dtype
        device = result.device

        memory_formats = {
            torch.contiguous_format,
            torch.channels_last,
            torch.channels_last_3d,
        }

        memory_format = None
        for query_format in memory_formats:
            if result.is_contiguous(memory_format=query_format):
                memory_format = query_format
                break

        return TensorMetadata(shape, dtype, device, memory_format)

class ShapePropagation(fx.Interpreter):
    def run_node(self, n : fx.Node) -> Any:
        args = self.map_nodes_to_values(n.args, n)
        in_metadata = [TensorMetadata.extract_metadata(a) for a in args if isinstance(a, torch.Tensor)]

        if len(in_metadata) > 0:
            n.in_metadata = in_metadata

        output = super().run_node(n)

        output_tuple = output if isinstance(output, tuple) else tuple((output,))
        out_metadata = [TensorMetadata.extract_metadata(a) for a in output_tuple if isinstance(a, torch.Tensor)]
        if len(out_metadata) > 0:
            n.out_metadata = out_metadata

        return output
