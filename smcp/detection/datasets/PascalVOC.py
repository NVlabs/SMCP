# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

import PIL.Image as im
import pytorch_lightning as pl
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
import torchvision.transforms.functional as F
import xml.etree.ElementTree as ET

from smcp.detection.datasets.augmentations import photometric_distort, random_crop, resize, expand, flip

def get_VOC_label_map() -> Dict[str, int]:
    voc_labels = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    label_map = { k: v + 1 for v, k in enumerate(voc_labels) }
    label_map["background"] = 0

    return label_map

class ImageOnlyPascalVOCDataset(Dataset):
    def __init__(
        self, image_paths: List[str], transform: Callable[..., torch.Tensor],
    ):
        """
        :param image_paths: list of paths to images
        :param transform: transformation function
        """
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, i: int) -> torch.Tensor:
        # Read image
        image = im.open(self.image_paths[i], mode="r").convert("RGB")
        image = F.to_tensor(image)

        return self.transform(image)

    def __len__(self) -> int:
        return len(self.image_paths)

    def collate_fn(self, batch: torch.Tensor):
        """
        :param batch: an iterable of N images from __getitem__()
        :return: a tensor of images
        """
        return default_collate(batch).to(memory_format=torch.channels_last)

class Annotation(TypedDict):
    boxes: List[List[int]]
    labels: List[int]
    difficulties: List[int]

PascalVOCDataBatch = Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]

class PascalVOCDataset(ImageOnlyPascalVOCDataset):
    def __init__(
        self, image_paths: List[str], objects: List[Annotation], transform: Callable[..., torch.Tensor],
        keep_difficult: bool = False
    ):
        """
        :param image_paths: list of paths to images
        :param split: split, one of "train" or "test"
        :param transform: transformation function
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        super().__init__(image_paths, transform)

        self.objects = objects
        self.keep_difficult = keep_difficult

        assert len(self.image_paths) == len(self.objects)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Read image
        image = im.open(self.image_paths[i], mode="r").convert("RGB")
        image = F.to_tensor(image)

        # Read objects in this image (bounding boxes, labels, difficulties)
        object = self.objects[i]
        boxes = torch.tensor(object["boxes"], dtype=torch.float32)  # (n_objects, 4)
        labels = torch.tensor(object["labels"], dtype=torch.long)  # (n_objects)
        difficulties = torch.tensor(object["difficulties"], dtype=torch.bool)  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        return self.transform(image, boxes, labels, difficulties)

    def __len__(self) -> int:
        return len(self.image_paths)

    def collate_fn(
        self, batch: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> PascalVOCDataBatch:
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images: List[torch.Tensor] = []
        boxes: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        difficulties: List[torch.Tensor] = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = default_collate(images).to(memory_format=torch.channels_last)

        return images, boxes, labels, difficulties  # tensor (N, 3, *, *), 3 lists of N tensors each

def parse_annotation(label_map: Dict[str, int], annotation_path: str) -> Annotation:
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter("object"):

        difficult = int(object.find("difficult").text == "1")

        label = object.find("name").text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text) - 1
        ymin = int(bbox.find("ymin").text) - 1
        xmax = int(bbox.find("xmax").text) - 1
        ymax = int(bbox.find("ymax").text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}

def create_imageonly_transform(dims: Tuple[int, int], dtype: torch.dtype, random_noise: bool = False, noise_scale_factor: float = 0) -> Callable:
    normalizer = imagenet_normalization()
    dtype_transform = transforms.ConvertImageDtype(dtype)

    def imageonly_transform(image: torch.Tensor) -> torch.Tensor:
        # Resize image. Also convert absolute boundary coordinates to their fractional form
        image = resize(image, None, dims, only_image=True)

        # Add Gaussian random noise
        if random_noise:
            noise = torch.randn_like(image) * noise_scale_factor
            image = torch.add(image, noise)
            image = torch.clamp(image, 0., 1.)

        # Normalize by mean and standard deviation of ImageNet data
        image = normalizer(image)

        # Convert to correct type
        image = dtype_transform(image)

        return image

    return imageonly_transform

def create_train_transform(dims: Tuple[int, int], dtype: torch.dtype) -> Callable:
    normalizer = imagenet_normalization()
    dtype_transform = transforms.ConvertImageDtype(dtype)

    def train_transform(
        image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, difficulties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # A series of photometric distortions in random order, each with 50% chance of occurrence
        image = photometric_distort(image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            image, boxes = expand(image, boxes, filler=normalizer.mean)

        # Randomly crop image (zoom in)
        image, boxes, labels, difficulties = random_crop(image, boxes, labels, difficulties)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            image, boxes = flip(image, boxes)

        # Resize image. Also convert absolute boundary coordinates to their fractional form
        image, boxes = resize(image, boxes, dims=dims, only_image=False)

        # Normalize by mean and standard deviation of ImageNet data
        image = normalizer(image)

        # Convert to correct type
        image = dtype_transform(image)

        return image, boxes, labels, difficulties

    return train_transform

def create_test_transform(dims: Tuple[int, int], dtype: torch.dtype) -> Callable:
    normalizer = imagenet_normalization()
    dtype_transform = transforms.ConvertImageDtype(dtype)

    def test_transform(
        image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, difficulties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Resize image. Also convert absolute boundary coordinates to their fractional form
        image, boxes = resize(image, boxes, dims=dims, only_image=False)

        # Normalize by mean and standard deviation of ImageNet data
        image = normalizer(image)

        # Convert to correct type
        image = dtype_transform(image)

        return image, boxes, labels, difficulties

    return test_transform

class PascalVOCDataModule(pl.LightningDataModule):
    num_classes: int = 21

    def __init__(
        self,
        data_dir: str,
        dims: Tuple[int, int] = (300, 300),
        only_image: bool = False,
        keep_difficult: bool = False,
        random_noise: bool = False,
        random_std_scale_factor: float = 0.1,
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
            data_dir: path to the PascalVOC dataset files
            dims: output dimensions of the image
            only_image: whether to only yield the image
            keep_difficult: whether to keep difficult examples (always included in val)
            random_noise: whether to add Gaussian noise (only for image-only case)
            random_std_scale_factor: std of random noise to add (only for image-only case)
            num_workers: how many data workers
            batch_size: batch_size
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true, will drop last batch during training (if not full size)
            dtype: dtype to cast the image to
        """
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dtype = dtype

        self.only_image = only_image
        self.keep_difficult = keep_difficult

        self.label_map = get_VOC_label_map()

        # Define the transforms
        if self.only_image:
            imageonly_transform = create_imageonly_transform(dims, dtype, random_noise, random_std_scale_factor)
            self.train_transforms = imageonly_transform
            self.val_transforms = imageonly_transform
            self.test_transforms = imageonly_transform
        else:
            self.train_transforms = create_train_transform(dims, dtype)
            self.test_transforms = self.val_transforms = create_test_transform(dims, dtype)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets with lists of images, the bounding boxes and labels of the objects in these images.
        """
        voc07_path = os.path.abspath(os.path.join(self.data_dir, "VOC2007"))
        voc12_path = os.path.abspath(os.path.join(self.data_dir, "VOC2012"))

        # Training data
        train_07_image_paths, train_07_objects = self._load_paths_and_objects_from_folder(voc07_path, "trainval")
        train_12_image_paths, train_12_objects = self._load_paths_and_objects_from_folder(voc12_path, "trainval")

        train_image_paths = train_07_image_paths + train_12_image_paths
        train_objects = train_07_objects + train_12_objects

        n_objects = sum(len(obj["boxes"]) for obj in train_objects)
        print(f"There are {len(train_image_paths)} training images containing a total of {n_objects} objects.")

        if self.only_image:
            self.train_dataset = ImageOnlyPascalVOCDataset(train_image_paths, self.train_transforms)
        else:
            self.train_dataset = PascalVOCDataset(train_image_paths, train_objects, self.train_transforms, self.keep_difficult)

        # Validation data
        val_07_image_paths, val_07_objects = self._load_paths_and_objects_from_folder(voc07_path, "val")
        val_12_image_paths, val_12_objects = self._load_paths_and_objects_from_folder(voc12_path, "val")

        val_image_paths = val_07_image_paths + val_12_image_paths
        val_objects = val_07_objects + val_12_objects

        n_objects = sum(len(obj["boxes"]) for obj in val_objects)
        print(f"There are {len(val_image_paths)} validation images containing a total of {n_objects} objects.")

        if self.only_image:
            self.val_dataset = ImageOnlyPascalVOCDataset(val_image_paths, self.val_transforms)
        else:
            self.val_dataset = PascalVOCDataset(val_image_paths, val_objects, self.val_transforms, keep_difficult=True)

        # Test data
        test_image_paths, test_objects = self._load_paths_and_objects_from_folder(voc07_path, "test")

        n_objects = sum(len(obj["boxes"]) for obj in test_objects)
        print(f"There are {len(test_image_paths)} test images containing a total of {n_objects} objects.")

        if self.only_image:
            self.test_dataset = ImageOnlyPascalVOCDataset(test_image_paths, self.test_transforms)
        else:
            self.test_dataset = PascalVOCDataset(test_image_paths, test_objects, self.test_transforms, keep_difficult=True)

    def _load_paths_and_objects_from_folder(self, folder_path: str, type: str) -> Tuple[List[str], List[Annotation]]:
        image_paths: List[str] = []
        all_objects: List[Annotation] = []

        # Find IDs of images in the test data
        with open(os.path.join(folder_path, f"ImageSets/Main/{type}.txt")) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation"s XML file
            objects = parse_annotation(self.label_map, os.path.join(folder_path, "Annotations", f"{id}.xml"))
            num_boxes = len(objects["boxes"])
            if num_boxes == 0:
                continue

            all_objects.append(objects)
            image_paths.append(os.path.join(folder_path, "JPEGImages", f"{id}.jpg"))

        assert len(image_paths) == len(all_objects)

        return image_paths, all_objects

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory, prefetch_factor=6,
            collate_fn=self.train_dataset.collate_fn, drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, prefetch_factor=6,
            collate_fn=self.val_dataset.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, prefetch_factor=6,
            collate_fn=self.test_dataset.collate_fn
        )
