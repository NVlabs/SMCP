# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# Adapted from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
# and https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py

from typing import List
import warnings

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, _LRScheduler

class _WarmupMixin():
    def _get_warmup_lr(self) -> List[float]:
        if self.last_epoch == 0:
            factor = 1 / (self.warmup_length + 1)
            return [base_lr * factor for base_lr in self.base_lrs]
        elif self.last_epoch <= self.warmup_length:
            return [
                group["lr"] + base_lr / (self.warmup_length + 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        raise NotImplementedError

class WarmupMultiStepLR(MultiStepLR, _WarmupMixin):
    def __init__(
        self, optimizer: Optimizer, warmup_length: int,
        milestones: List[int], gamma: float, last_epoch: int = -1, verbose: bool = False
    ):
        self.warmup_length = warmup_length
        super().__init__(optimizer, milestones, gamma, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.warmup_length:
            return self._get_warmup_lr()

        return super().get_lr()

class WarmupCustomMultiStepLR(_LRScheduler, _WarmupMixin):
    def __init__(
        self, optimizer: Optimizer, warmup_length: int,
        milestones: List[int], gammas: List[float], last_epoch: int = -1, verbose: bool = False
    ):
        """
        Allows different gamma drop at each milestone
        """
        self.warmup_length = warmup_length
        self.gamma_dict = { e: g for (e, g) in zip(milestones, gammas)}

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.warmup_length:
            return self._get_warmup_lr()
        elif self.last_epoch not in self.gamma_dict:
            return [group['lr'] for group in self.optimizer.param_groups]

        gamma = self.gamma_dict[self.last_epoch]

        return [group['lr'] * gamma for group in self.optimizer.param_groups]

class WarmupLinearLR(_LRScheduler, _WarmupMixin):
    def __init__(
        self, optimizer: Optimizer, warmup_length: int, total_epochs: int,
        last_epoch: int = -1, verbose: bool = False
    ):
        self.warmup_length = warmup_length
        self.decay_length = total_epochs - warmup_length
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.warmup_length:
            return self._get_warmup_lr()

        return [
            group['lr'] - base_lr / self.decay_length
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

class WarmupCosineLR(_LRScheduler, _WarmupMixin):
    def __init__(
        self, optimizer: Optimizer, warmup_length: int, total_epochs: int,
        last_epoch: int = -1, verbose: bool = False
    ):
        self.warmup_length = warmup_length
        self.decay_length = total_epochs - warmup_length
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.warmup_length:
            return self._get_warmup_lr()

        old_e = self.last_epoch - 1 - self.warmup_length
        old_factor = 0.5 * (1 + np.cos(np.pi * old_e / self.decay_length))
        e = old_e + 1
        factor = 0.5 * (1 + np.cos(np.pi * e / self.decay_length))

        return [
            group["lr"] * factor / old_factor
            for group in self.optimizer.param_groups
        ]

class WarmupExponentialLR(ExponentialLR, _WarmupMixin):
    def __init__(
        self, optimizer: Optimizer, warmup_length: int,
        gamma: float, last_epoch: int = -1, verbose: bool = False
    ):
        self.warmup_length = warmup_length
        super().__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.warmup_length:
            return self._get_warmup_lr()

        return super().get_lr()
