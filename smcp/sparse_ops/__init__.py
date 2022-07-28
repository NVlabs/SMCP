# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from smcp.sparse_ops.base_pruning import PruningSchedule
from smcp.sparse_ops.channel_pruning import ChannelBNRescalingType, ChannelPruning, ChannelPruningSchedule, ChannelPruningType
from smcp.sparse_ops.channel_structure import ChannelStructure
from smcp.sparse_ops.channel_costing import CostingType, create_coster
from smcp.sparse_ops.dynamic_pruning import DynamicPruning, PruningLogVerbosity
from smcp.sparse_ops.importance import ImportanceType, ImportanceGradType, ImportanceHessType
from smcp.sparse_ops.importance_accumulator import create_importance_accumulator, ImportanceAccumulatorType
from smcp.sparse_ops.parameter_masking import ParameterMaskingType

__all__ = [
    # methods
    "create_coster",
    "create_importance_accumulator",

    # classes/types
    "ChannelBNRescalingType",
    "ChannelPruning",
    "ChannelPruningSchedule",
    "ChannelPruningType",
    "ChannelStructure",
    "CostingType",
    "DynamicPruning",
    "ImportanceAccumulatorType",
    "ImportanceType",
    "ImportanceGradType",
    "ImportanceHessType",
    "ParameterMaskingType",
    "PruningLogVerbosity",
    "PruningSchedule"
]
