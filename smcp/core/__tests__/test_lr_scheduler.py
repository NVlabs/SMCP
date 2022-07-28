# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import math
import pytest
from torch import optim

from smcp.core.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR, WarmupCustomMultiStepLR, WarmupLinearLR, WarmupExponentialLR

def test_no_warmup() -> None:
    base_lr = 1e-2
    optimizer = optim.SGD([
        {"params": []},
        {"params": [], "lr": 0.1 * base_lr }
    ], base_lr, 1, 0, 0, False)

    warmup_length = 0
    total_epochs = 50
    sch = WarmupLinearLR(optimizer, warmup_length, total_epochs)

    lrs = []
    for _ in range(total_epochs):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        assert pytest.approx(0.1 * lr) == optimizer.param_groups[1]["lr"]

        sch.step()

    assert pytest.approx(lrs) == [1e-2 * p/50 for p in range(50, 0, -1)]

def test_warmup_linear() -> None:
    base_lr = 1e-2
    optimizer = optim.SGD([
        {"params": []},
        {"params": [], "lr": 0.1 * base_lr }
    ], base_lr, 1, 0, 0, False)

    warmup_length = 4
    total_epochs = 54
    sch = WarmupLinearLR(optimizer, warmup_length, total_epochs)

    lrs = []
    for _ in range(total_epochs):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        assert pytest.approx(0.1 * lr) == optimizer.param_groups[1]["lr"]

        sch.step()

    assert lrs[:4] == [2e-3, 4e-3, 6e-3, 8e-3]
    assert pytest.approx(lrs[4:]) == [1e-2 * p/50 for p in range(50, 0, -1)]

def test_warmup_multistep() -> None:
    base_lr = 1e-2
    optimizer = optim.SGD([
        {"params": []},
        {"params": [], "lr": 0.1 * base_lr }
    ], base_lr, 1, 0, 0, False)

    warmup_length = 4
    milestones = [10, 20, 40]
    total_epochs = 54
    sch = WarmupMultiStepLR(optimizer, warmup_length, milestones, 0.1)

    lrs = []
    for _ in range(total_epochs):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        assert pytest.approx(0.1 * lr) == optimizer.param_groups[1]["lr"]

        optimizer.step()
        sch.step()

    assert lrs[:4] == [2e-3, 4e-3, 6e-3, 8e-3]
    assert all(r == 1e-2 for r in lrs[4:10])
    assert all(r == 1e-3 for r in lrs[10:20])
    assert all(r == 1e-4 for r in lrs[20:40])
    assert all(r == 1e-5 for r in lrs[40:])

def test_warmup_custom_multistep() -> None:
    base_lr = 1e-2
    optimizer = optim.SGD([
        {"params": []},
        {"params": [], "lr": 0.1 * base_lr }
    ], base_lr, 1, 0, 0, False)

    warmup_length = 4
    milestones = [10, 20, 26, 28]
    gammas = [3/8, 1/3, 2/5, 1/10]
    total_epochs = 30
    sch = WarmupCustomMultiStepLR(optimizer, warmup_length, milestones, gammas)

    lrs = []
    for _ in range(total_epochs):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        assert pytest.approx(0.1 * lr) == optimizer.param_groups[1]["lr"]

        optimizer.step()
        sch.step()

    assert lrs[:4] == [2e-3, 4e-3, 6e-3, 8e-3]
    assert all(r == 1e-2 for r in lrs[4:10])
    assert all(r == 3.75e-3 for r in lrs[10:20])
    assert all(pytest.approx(r) == 1.25e-3 for r in lrs[20:26])
    assert all(pytest.approx(r) == 5e-4 for r in lrs[26:28])
    assert all(pytest.approx(r) == 5e-5 for r in lrs[28:])

def test_warmup_exponential() -> None:
    base_lr = 1e-2
    optimizer = optim.SGD([
        {"params": []},
        {"params": [], "lr": 0.1 * base_lr }
    ], base_lr, 1, 0, 0, False)

    warmup_length = 4
    total_epochs = 54
    sch = WarmupExponentialLR(optimizer, warmup_length, gamma=0.95)

    lrs = []
    for _ in range(total_epochs):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        assert pytest.approx(0.1 * lr) == optimizer.param_groups[1]["lr"]

        sch.step()

    assert lrs[:4] == [2e-3, 4e-3, 6e-3, 8e-3]
    assert pytest.approx(lrs[4:]) == [1e-2 * 0.95**p for p in range(50)]

def test_warmup_cosine() -> None:
    base_lr = 1e-2
    optimizer = optim.SGD([
        {"params": []},
        {"params": [], "lr": 0.1 * base_lr }
    ], base_lr, 1, 0, 0, False)

    warmup_length = 4
    total_epochs = 54
    sch = WarmupCosineLR(optimizer, warmup_length, total_epochs)

    lrs = []
    for _ in range(total_epochs):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        assert pytest.approx(0.1 * lr) == optimizer.param_groups[1]["lr"]

        sch.step()

    assert lrs[:4] == [2e-3, 4e-3, 6e-3, 8e-3]
    assert pytest.approx(lrs[4:]) == [1e-2 * 0.5 * (1 + math.cos(math.pi * p / 50)) for p in range(50)]
