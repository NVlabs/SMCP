# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR100

from smcp.core.image_collate import create_image_collate

class UpscaledCIFAR10DataModule(CIFAR10DataModule):
    def __init__(
        self, *args, image_size: int = 224,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dtype = dtype

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            cifar10_normalization(),
            transforms.ConvertImageDtype(self.dtype)
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.image_size + 32),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            cifar10_normalization(),
            transforms.ConvertImageDtype(self.dtype)
        ])

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=create_image_collate(torch.channels_last)
        )

class UpscaledCIFAR100DataModule(UpscaledCIFAR10DataModule):
    name = "cifar100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def num_classes(self) -> int:
        return 100
