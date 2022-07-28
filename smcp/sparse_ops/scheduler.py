# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

def _exp_prune_schedule(total_epochs: int, prune_ratio: float) -> torch.Tensor:
    b = 0.05
    sch = 1 - torch.exp(-b * torch.arange(total_epochs))

    return prune_ratio * sch / sch[-1]

# exp schedule proposed in FORCE (https://arxiv.org/pdf/2006.09081.pdf)
def _exp2_prune_schedule(total_epochs: int, prune_ratio: float) -> torch.Tensor:
    alpha = torch.arange(total_epochs) / (total_epochs - 1)
    sch = 1 - (1 - prune_ratio) ** alpha

    return sch

def _linear_prune_schedule(total_epochs: int, prune_ratio: float) -> torch.Tensor:
    return prune_ratio * torch.arange(total_epochs) / (total_epochs - 1)

def _prune_schedule(schedule: str, total_epochs: int, prune_ratio: float) -> torch.Tensor:
    """
    Assumes total_epochs >=2. First value will be 0. Last value will be `prune_ratio`.
    """

    if schedule == "exp":
        return _exp_prune_schedule(total_epochs, prune_ratio)
    elif schedule == "exp2":
        return _exp2_prune_schedule(total_epochs, prune_ratio)
    else:
        return _linear_prune_schedule(total_epochs, prune_ratio)

def get_prune_schedule(
    schedule: str, prune_ratio: float, total_epochs: int,
    warmup_length: int, cooldown_length: int,
) -> torch.Tensor:
    """
    Schedule the target number of neurons to prune by each epoch

    Args:
    -----
        schedule: Type of pruning schedule to use. Choose from "exp", "exp2", and "linear".
        prune_ratio: Fraction to prune
        total_epochs: Number of total training epochs
        warmup_length: Number of epochs with no pruning.
        cooldown_length: Number of epochs with full `prune_ratio` pruning.

    Returns:
    --------
        sch: Array with size of total epochs. Each item is the cumulative ratio of neurons to prune by the corresponding epoch.
    """

    num_prune_epochs = total_epochs - warmup_length - cooldown_length

    assert warmup_length >= 0, "Schedule warmup length must be non-negative"
    assert cooldown_length >= 1, "Schedule cooldown length must be at least 1"
    assert num_prune_epochs >= 0, "Schedule warmup and cooldown lengths are bigger than the total number of epochs"

    # Get the schedule
    sch = torch.zeros(total_epochs)
    sch[warmup_length:-cooldown_length] = _prune_schedule(schedule, num_prune_epochs + 2, prune_ratio)[1:-1]
    sch[-cooldown_length:] = prune_ratio

    return sch
