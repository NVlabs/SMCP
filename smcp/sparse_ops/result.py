# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import annotations
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import reduce
from typing import Dict, Tuple, Type

import torch

@torch.jit.script
def get_mask_stats(new_mask: torch.Tensor, old_mask: torch.Tensor) -> Tuple[int, int, int, int]:
    numel = new_mask.numel()
    total_pruned = numel - new_mask.sum()
    new_pruned = torch.logical_and(~new_mask, old_mask).sum()
    unpruned = torch.logical_and(new_mask, ~old_mask).sum()

    return numel, total_pruned.item(), new_pruned.item(), unpruned.item()

class PruneResult(ABC):
    @abstractmethod
    def todict(self, prefix: str = "", include_layer_info: bool = False) -> Dict[str, float]:
        pass

@dataclass
class LayerPruneResult(PruneResult):
    numel: int
    total_pruned: int
    new_pruned: int
    unpruned: int
    num_changed_since_start: int

    def __init__(self, numel: int = 0, total_pruned: int = 0, new_pruned: int = 0, unpruned: int = 0, num_changed_since_start: int = 0):
        self.numel = numel
        self.total_pruned = total_pruned
        self.new_pruned = new_pruned
        self.unpruned = unpruned
        self.num_changed_since_start = num_changed_since_start

    def __add__(self, other: LayerPruneResult) -> LayerPruneResult:
        if not isinstance(other, LayerPruneResult):
            raise NotImplementedError(f"LayerPruneResult cannot be added with type {type(other)}")

        numel = self.numel + other.numel
        total_pruned = self.total_pruned + other.total_pruned
        new_pruned = self.new_pruned + other.new_pruned
        unpruned = self.unpruned + other.unpruned
        num_changed_since_start = self.num_changed_since_start + other.num_changed_since_start

        return LayerPruneResult(numel, total_pruned, new_pruned, unpruned, num_changed_since_start)

    def __mul__(self, other: float) -> LayerPruneResult:
        if not isinstance(other, (int, float)):
            raise NotImplementedError(f"LayerPruneResult cannot be multplied by type {type(other)}")

        numel = self.numel * other
        total_pruned = self.total_pruned * other
        new_pruned = self.new_pruned * other
        unpruned = self.unpruned * other
        num_changed_since_start = self.num_changed_since_start * other

        return LayerPruneResult(numel, total_pruned, new_pruned, unpruned, num_changed_since_start)

    __rmul__ = __mul__

    @classmethod
    def from_masks(cls: Type[LayerPruneResult], new_mask: torch.Tensor, old_mask: torch.Tensor) -> LayerPruneResult:
        return cls(*get_mask_stats(new_mask, old_mask), 0)

    @property
    def frac_pruned(self) -> float:
        return self._frac_elements(self.total_pruned)

    @property
    def frac_new_pruned(self) -> float:
        return self._frac_elements(self.new_pruned)

    @property
    def frac_unpruned(self) -> float:
        return self._frac_elements(self.unpruned)

    @property
    def frac_changed(self) -> float:
        return self._frac_elements(self.num_changed_since_start)

    def _frac_elements(self, value: int) -> float:
        return value / self.numel if self.numel > 0 else 0

    def __repr__(self) -> str:
        return f"Total num: {self.numel}. Num pruned: {self.total_pruned}. Num newly pruned: {self.new_pruned}. Num un-pruned: {self.unpruned}."

    def todict(self, prefix: str = "", include_layer_info: bool = False) -> Dict[str, float]:
        return {
            f"{prefix}numel": self.numel,
            f"{prefix}frac_pruned": self.frac_pruned,
            f"{prefix}frac_new_pruned": self.frac_new_pruned,
            f"{prefix}frac_unpruned": self.frac_unpruned,
            f"{prefix}frac_changed_since_start": self.frac_changed,
        }

class NetworkPruneResult(PruneResult):
    layer_results: Dict[str, LayerPruneResult]
    total_result: LayerPruneResult

    def __init__(self, layer_results: Dict[str, LayerPruneResult]):
        self.layer_results = layer_results
        self.total_result = sum(layer_results.values(), LayerPruneResult())

    def todict(self, prefix: str = "", include_layer_info: bool = False) -> Dict[str, float]:
        total_dict = self.total_result.todict(prefix)

        if include_layer_info:
            layer_dicts = [
                layer_result.todict(f"{prefix}{layer}/")
                for layer, layer_result in self.layer_results.items()
            ]
            layer_dict = reduce(lambda a,b: {**a, **b}, layer_dicts, {})

            return { **total_dict, **layer_dict }

        return total_dict
