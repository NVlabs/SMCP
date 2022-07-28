# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

@torch.jit.script
def label_smoothing(x: torch.Tensor, target: torch.Tensor, smoothing: float) -> torch.Tensor:
    logprobs = F.log_softmax(x, dim=-1)

    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss

    return loss.mean()

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing: float = 0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return label_smoothing(x, target, self.smoothing)
