# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse

import torch

from smcp.classification.datasets import UpscaledCIFAR10DataModule, UpscaledCIFAR100DataModule, ImagenetDataModule
from smcp.classification.models import get_classification_model
from smcp.core.model_summary import time_inference, model_summary
from smcp.sparse_ops.model_clean import clean_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for testing inference speed")
    parser.add_argument("path", type=str, help="Path to model")
    parser.add_argument("-a", "--arch", type=str, default=None)

    # Inference options
    parser.add_argument("--slim", action="store_true", default=False,
                        help="Whether to slim the model before inference")
    parser.add_argument("--fuse", action="store_true", default=False,
                        help="Whether to fuse Conv-BN layers before inference")
    parser.add_argument("--torchscript", action="store_true", default=False,
                        help="Whether to convert to torchscript before inference")
    parser.add_argument("--inf-warmup", type=int, default=10,
                        help="Number of warmup batches before timing inference")
    parser.add_argument("--inf-batches", type=int, default=30,
                        help="Number of batches to perform timing over")

    # Dataset options
    parser.add_argument("--dataset", type=str, default="Imagenet",
                        help="The name of the dataset")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="The root directory of the dataset")
    parser.add_argument("--workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="The number of examples in one training batch")

    args = parser.parse_args()

    device = torch.device("cuda")

    # Setup datamodule
    dm_cls = None
    if args.dataset == "Imagenet":
        dm_cls = ImagenetDataModule
    elif args.dataset == "CIFAR10":
        dm_cls = UpscaledCIFAR10DataModule
    elif args.dataset == "CIFAR100":
        dm_cls = UpscaledCIFAR100DataModule
    else:
        raise NotImplementedError(f"Dataset {args.dataset} unknown")

    dm = dm_cls(
        args.data_root, num_workers=args.workers, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=True, dtype=torch.float32
    )
    dm.prepare_data()
    dm.setup()
    dataloader = dm.val_dataloader()

    # Load model
    if args.arch is None:
        model = torch.load(args.path, map_location=device)
    else:
        model = get_classification_model(args.arch, dm.num_classes, pretrained=args.path)

    model = model.to(device=device, memory_format=torch.channels_last)
    model.eval()

    example_input, _ = next(iter(dataloader))
    model = clean_model(model, example_input.to(device), args.slim, args.fuse, args.torchscript)

    # Run inference
    avg_batch_time = time_inference(model, dataloader)

    print("Infer time (ms/image)", avg_batch_time / args.batch_size)
    print("FPS:", 1e3 * args.batch_size  / avg_batch_time)

    # Get summary
    macs, params = model_summary(model, dataloader)

    print("Params(M)", params)
    print("MACs(G)", macs)
