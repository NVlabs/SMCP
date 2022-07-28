# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from abc import abstractmethod
from enum import Enum
from typing import Dict, Optional, TypeVar

import torch

class ImportanceAccumulatorType(Enum):
    Latest = "Latest"
    Average = "Average"
    Momentum = "Momentum"

TKey = TypeVar("TKey")
class ImportanceAccumulator(Dict[TKey, torch.Tensor]):
    def should_update(self, is_about_to_prune: bool) -> bool:
        return True

    @abstractmethod
    def update(self, name: TKey, importance: torch.Tensor) -> None:
        pass

class LatestAccumulator(ImportanceAccumulator[TKey]):
    def should_update(self, is_about_to_prune: bool) -> bool:
        return is_about_to_prune

    def update(self, name: TKey, importance: torch.Tensor) -> None:
        self.__setitem__(name, importance)

class AveragingAccumulator(ImportanceAccumulator[TKey]):
    def __init__(self):
        super().__init__()

        self.num_accums = 0

    def update(self, name: TKey, importance: torch.Tensor) -> None:
        self.num_accums += 1

        if self.num_accums > 0:
            past = self.__getitem__(name)
            importance = past + (importance - past) / self.num_accums

        self.__setitem__(name, importance)

class MomentumAccumulator(ImportanceAccumulator[TKey]):
    def __init__(self, momentum: float):
        super().__init__()

        self.momentum = momentum

    def update(self, name: TKey, importance: torch.Tensor) -> None:
        if self.__contains__(name):
            past = self.__getitem__(name)
            importance = self.momentum * past + (1 - self.momentum) * importance
            # NOTE: There is no bias correction here

        self.__setitem__(name, importance)

def create_importance_accumulator(
    accumulator_type: ImportanceAccumulatorType, momentum: Optional[float] = 0.9
) -> ImportanceAccumulator:
    if accumulator_type == ImportanceAccumulatorType.Latest:
        return LatestAccumulator()
    elif accumulator_type == ImportanceAccumulatorType.Average:
        return AveragingAccumulator()
    elif accumulator_type == ImportanceAccumulatorType.Momentum:
        assert momentum is not None, f"Momentum accumulator requires the momentum value to be set"

        return MomentumAccumulator(momentum)

    raise NotImplementedError(f"Accumulator type {ImportanceAccumulatorType} invalid")
