# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import List, Tuple

import torch
from torch.utils.data._utils.collate import default_collate

def create_image_collate(memory_format: torch.memory_format = torch.contiguous_format):
    def image_collate(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = default_collate(batch)

        return x.to(memory_format=memory_format), target

    return image_collate
