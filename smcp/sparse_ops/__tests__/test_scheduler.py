# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import pytest
import torch

from smcp.sparse_ops.scheduler import get_prune_schedule

@pytest.mark.parametrize("schedule", ["exp", "exp2", "linear"])
@pytest.mark.parametrize(
    "warmup,cooldown",
    [(0, 1), (1, 1), (1, 2), (2, 2), (10, 10), (88, 1), (89, 1), (0, 89), (1, 89), (0, 90)]
)
def test_prune_schedule(schedule: str, warmup: int, cooldown: int):
    prune_ratio = 0.5
    total_epochs = 90

    schedule = get_prune_schedule(schedule, prune_ratio, total_epochs, warmup, cooldown)

    assert torch.isfinite(schedule).all(), "Schedule has NaN(s)"
    assert torch.all(schedule[:warmup] == 0), "Schedule is not zero for warmup"
    assert schedule[warmup] > 0, "Schedule is zero for first non-warmup epoch"

    if cooldown < total_epochs:
        assert schedule[-(cooldown+1)] < prune_ratio, "Schedule is prune_ratio for last non-cooldown epoch"
    assert torch.all(schedule[-cooldown:] == prune_ratio), "Schedule is not prune_ratio for cooldown"
    assert torch.all(schedule[1:] - schedule[:-1] >= 0), "Schedule is not monotonically non-decreasing"
