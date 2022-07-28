# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from smcp.classification.datasets.cifar import UpscaledCIFAR10DataModule, UpscaledCIFAR100DataModule
from smcp.classification.datasets.imagenet import ImagenetDataModule

__all__ = [
    "UpscaledCIFAR10DataModule",
    "UpscaledCIFAR100DataModule",
    "ImagenetDataModule"
]
