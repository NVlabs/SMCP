# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Any

import os
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from smcp.core.image_collate import create_image_collate

class ImagenetDataModule(pl.LightningDataModule):
    num_classes: int = 1000

    def __init__(
        self,
        data_dir: str,
        num_workers: int = 8,
        batch_size: int = 128,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        dtype: torch.dtype = torch.float32,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            num_workers: how many data workers
            batch_size: batch_size
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true, will drop last batch during training (if not full size)
            dtype: dtype to cast the image to
        """
        super().__init__(*args, **kwargs)

        self.dims = (3, 224, 224)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dtype = dtype

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization(),
            transforms.ConvertImageDtype(self.dtype)
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_normalization(),
            transforms.ConvertImageDtype(self.dtype)
        ])

    def train_dataloader(self) -> DataLoader:
        traindir = os.path.join(self.data_dir, "train")
        train_dataset = datasets.ImageFolder(
            traindir,
            self.train_transforms
        )

        return DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory, prefetch_factor=6,
            collate_fn=create_image_collate(torch.channels_last), drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader:
        valdir = os.path.join(self.data_dir, "val")
        val_dataset = datasets.ImageFolder(
            valdir,
            self.val_transforms
        )

        return DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, prefetch_factor=6,
            collate_fn=create_image_collate(torch.channels_last)
        )
