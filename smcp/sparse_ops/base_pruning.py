# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from torch import nn

from smcp.sparse_ops.importance import ImportanceFunction
from smcp.sparse_ops.importance_accumulator import ImportanceAccumulator
from smcp.sparse_ops.parameter_masking import add_masking, ParameterMaskingType, remove_masking
from smcp.sparse_ops.result import PruneResult

MaskConvInfo = Tuple[torch.Tensor, torch.Tensor]

class PruningSchedule:
    def __init__(
        self, num_epochs: int, warmup_length: int, cooldown_length: int, rewiring_freq: int,
    ) -> None:
        """
        num_epochs: number of epochs
        warmup_length: number of epochs at the beginning to skip pruning
        cooldown_length: number of epochs at the end to skip pruning
        rewiring_freq: number of optimizer steps before rewiring the sparsity
        """
        self.num_epochs = num_epochs
        self.warmup_length = warmup_length
        self.cooldown_length = cooldown_length
        self.rewiring_freq = rewiring_freq

        assert num_epochs > warmup_length + cooldown_length, "Warmup + cooldown exceeds total number of epochs"

    def _should_prune_during_epoch(self, epoch: int) -> bool:
        return (epoch >= self.warmup_length) \
            and (epoch < (self.num_epochs - self.cooldown_length))

    def should_prune(self, epoch: int, global_step: int) -> bool:
        return self._should_prune_during_epoch(epoch) \
            and (global_step % self.rewiring_freq) == 0

class BasePruningMethod(ABC):
    _last_changed_dict: Dict[str, torch.Tensor]

    def __init__(
        self, masking_type: ParameterMaskingType, schedule: PruningSchedule,
        importance_accumulator: ImportanceAccumulator, track_mask_convergence: bool = False
    ) -> None:
        """
        Model pruning base.

        Args:
            masking_type: what kind of masking to perform (hard, soft, etc.)
            schedule: pruning schedule
            importance_accumulator: accumulates importance across steps
            track_mask_convergence: whether to track mask convergence
        """

        self.masking_type = masking_type
        self.track_mask_convergence = track_mask_convergence

        self._schedule = schedule
        self._importance_accumulator = importance_accumulator
        self._last_changed_dict = {}

    def should_mask(self, name: str, layer: nn.Module) -> bool:
        return True

    def apply_masking(self, model: nn.Module) -> None:
        for name, layer in model.named_modules():
            if self.should_mask(name, layer):
                add_masking(layer, self.masking_type)

    def remove_masking(self, model: nn.Module) -> None:
        model.apply(remove_masking)

    def should_update_importance(self, epoch: int, global_step: int) -> bool:
        does_prune_during_epoch = self._schedule._should_prune_during_epoch(epoch)
        is_about_to_prune = self._schedule.should_prune(epoch, global_step)

        return does_prune_during_epoch and self._importance_accumulator.should_update(is_about_to_prune)

    @abstractmethod
    def update_importance(self, model: nn.Module, importance_fn: ImportanceFunction) -> None:
        pass

    def reset_importance(self) -> None:
        self._importance_accumulator.clear()

    def should_prune(self, epoch: int, global_step: int) -> bool:
        return self._schedule.should_prune(epoch, global_step)

    @abstractmethod
    def prune(self, model: nn.Module, epoch: int, global_step: int) -> PruneResult:
        pass

    # Track when masks change
    # Enables mask convergence analysis as in Fig 4 of https://arxiv.org/pdf/2006.07253.pdf
    # Returns number of masks changed since the start
    @torch.no_grad()
    def _track_mask_changes(self, global_step: int, name: str, new_mask: torch.Tensor, old_mask: torch.Tensor) -> float:
        if not self.track_mask_convergence:
            return 0

        # Update mask last updated tracking
        if name not in self._last_changed_dict:
            self._last_changed_dict[name] = torch.full_like(new_mask, -1, dtype=torch.long)

        self._last_changed_dict[name][new_mask != old_mask] = global_step

        return (self._last_changed_dict[name] >= 0).sum().item()

    @torch.no_grad()
    def get_mask_convergence(self) -> Tuple[MaskConvInfo, Dict[str, MaskConvInfo]]:
        if not self.track_mask_convergence:
            return (torch.tensor([0]), torch.tensor([0])), {}

        # Get convergence information from each layer
        layer_conv_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, last_changed in self._last_changed_dict.items():
            steps, counts = last_changed.unique(return_counts=True)

            if steps[0] != -1:
                steps = torch.cat([torch.tensor([-1], device=steps.device), steps])
                counts = torch.cat([torch.tensor([0], device=counts.device), counts])

            # Get number not converged by end of step
            num_not_converged = last_changed.numel() - counts.cumsum(dim=0)

            layer_conv_dict[name] = (steps, num_not_converged)

        # Build network-level convergence info from the layers
        steps = torch.cat([s for (s, _) in layer_conv_dict.values()]).unique()
        counts = torch.zeros_like(steps)

        for l_steps, l_counts in layer_conv_dict.values():
            idxs = (steps.view(1, -1) >= l_steps.view(-1, 1)).int().argmin(dim=0) - 1
            counts += l_counts[idxs]

        return (steps, counts), layer_conv_dict
