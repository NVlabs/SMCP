# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
from math import ceil
from typing import Dict, Tuple, Union
import warnings

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
import torch
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy

from smcp.core.enum_parse import EnumAction
from smcp.core.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR, WarmupLinearLR
from smcp.sparse_ops import create_coster, create_importance_accumulator, ChannelBNRescalingType, ChannelPruning, ChannelPruningSchedule, ChannelPruningType, CostingType, DynamicPruning, ImportanceAccumulatorType, ImportanceType, ImportanceGradType, ImportanceHessType, ParameterMaskingType, PruningLogVerbosity, PruningSchedule
from smcp.classification.datasets import UpscaledCIFAR10DataModule, UpscaledCIFAR100DataModule, ImagenetDataModule
from smcp.classification.models import get_classification_model
from smcp.classification.losses import LabelSmoothing

# Disable pl deprecations
warnings.simplefilter("ignore", LightningDeprecationWarning)

class ImageClassifierParams:
    arch: str
    pretrained: Union[bool, str]
    num_classes: int
    label_smoothing: float
    learning_rate: float
    momentum: float
    nesterov: bool
    weight_decay: float
    bn_weight_decay: float
    lr_schedule: str
    warmup: int
    epochs: int

class ImageClassifier(pl.LightningModule):
    hparams: ImageClassifierParams
    model: nn.Module

    def __init__(
        self, arch: str, num_classes: int, label_smoothing: float, pretrained: Union[bool, str],
        learning_rate: float, momentum: float, nesterov: bool, weight_decay: float, bn_weight_decay: float,
        lr_schedule: str, warmup: int, epochs: int, **kwargs
    ):
        """Image Classifier model

        Args:
            arch: type of classifier architecture
            num_classes: number of image classes
            label_smoothing: [0, 1) value for label smoothing
            pretrained: whether to use a pretrained network. If a string, path to the pretrained weights
            learning_rate: learning rate
            momentum: SGD momentum
            nesterov: whether to enable Nesterov momentum
            weight_decay: amount of weight decay for non-BatchNorm weights
            bn_weight_decay: amount of weight decay for BatchNorm weights
            lr_schedule: LR scheduler type
            warmup: LR scheduler linear warmup time
            epochs: total number of training epochs
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = get_classification_model(arch, num_classes, pretrained=pretrained).to(memory_format=torch.channels_last)
        self.example_input_array = torch.ones(1, 3, 224, 224).to(memory_format=torch.channels_last)

        if label_smoothing > 0.0:
            self.criterion = LabelSmoothing(label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        acc_metrics = MetricCollection({
            "top1": Accuracy(num_classes=num_classes, top_k=1),
            "top5": Accuracy(num_classes=num_classes, top_k=5)
        })
        self.train_acc_metrics = acc_metrics.clone(prefix="train/")
        self.val_acc_metrics = acc_metrics.clone(prefix="val/")

    def configure_optimizers(self) -> optim.Optimizer:
        parameters_for_optimizer = list(self.model.named_parameters())

        bn_params = [v for n, v in parameters_for_optimizer if "bn" in n]
        rest_params = [v for n, v in parameters_for_optimizer if not "bn" in n]
        optimizer = optim.SGD(
            [
                {"params": bn_params, "weight_decay": self.hparams.bn_weight_decay},
                {"params": rest_params, "weight_decay": self.hparams.weight_decay}
            ],
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov
        )

        lr_scheduler = None
        if self.hparams.lr_schedule == "step":
            lr_scheduler = WarmupMultiStepLR(optimizer, self.hparams.warmup, [30,60,80], 0.1)
        elif self.hparams.lr_schedule == "step_prune":
            lr_scheduler = WarmupMultiStepLR(optimizer, self.hparams.warmup, [10,20,30], 0.1)
        elif self.hparams.lr_schedule == "cosine":
            lr_scheduler = WarmupCosineLR(optimizer, self.hparams.warmup, self.hparams.epochs)
        elif self.hparams.lr_schedule == "linear":
            lr_scheduler = WarmupLinearLR(optimizer, self.hparams.warmup, self.hparams.epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: optim.Optimizer, optimizer_idx: int) -> None:
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, target = batch

        logits = self.forward(x)
        loss = self.criterion(logits, target)

        self.log("train/loss", loss, sync_dist=True)

        preds = nn.functional.softmax(logits, dim=1)
        acc_metrics = self.train_acc_metrics(preds, target)
        self.log_dict(acc_metrics, sync_dist=True)

        return { "loss": loss }

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, target = batch

        logits = self.forward(x)
        loss = self.criterion(logits, target)

        self.log("val/loss", loss, sync_dist=True)

        preds = nn.functional.softmax(logits, dim=1)
        acc_metrics = self.val_acc_metrics(preds, target)
        self.log_dict(acc_metrics, sync_dist=True)

        return loss


def main(hparams):
    # Interpret/modify the hparams
    using_gpu = hparams.gpus is not None
    data_dtype = torch.float16 if hparams.fp16 else torch.float32
    precision = 16 if hparams.fp16 else 32
    accum_grad_batches = hparams.simulated_batch_size // hparams.batch_size
    accelerator = "ddp" if using_gpu else None
    sync_batchnorm =  using_gpu and hparams.batch_size <= 32

    eff_batch_size = hparams.batch_size * accum_grad_batches * hparams.num_nodes * (hparams.gpus if using_gpu else 1)
    hparams.learning_rate *= eff_batch_size / 256
    hparams.rewiring_freq = ceil(256 * hparams.rewiring_freq / eff_batch_size)

    # Setup datamodule
    dm_cls = None
    if hparams.dataset == "Imagenet":
        dm_cls = ImagenetDataModule
    elif hparams.dataset == "CIFAR10":
        dm_cls = UpscaledCIFAR10DataModule
    elif hparams.dataset == "CIFAR100":
        dm_cls = UpscaledCIFAR100DataModule
    else:
        raise NotImplementedError(f"Dataset {hparams.dataset} unknown")

    dm = dm_cls(
        hparams.data_root, num_workers=hparams.workers, batch_size=hparams.batch_size,
        shuffle=True, pin_memory=using_gpu, drop_last=True, dtype=data_dtype
    )

    # Setup model
    model = ImageClassifier(num_classes=dm.num_classes, **vars(hparams))

    # Setup trainer
    pl.seed_everything(hparams.seed, workers=True)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=hparams.output_dir,
        name=f"image_classifier-{hparams.dataset}",
        default_hp_metric=False
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(
            filename="image_classifier-epoch{epoch}-val_loss{val/loss:.4f}-top1{val/top1:.4f}",
            mode="max",
            monitor="val/top1",
            auto_insert_metric_name=False,
            save_last=True,
            every_n_val_epochs=hparams.ckpt_freq
        )
    ]

    if hparams.prune:
        importance_accum = create_importance_accumulator(hparams.importance_accumulator)

        if hparams.channel_type is not ChannelPruningType.Skip:
            coster = create_coster(hparams.costing_type, hparams.costing_latency_table)

            pruning_schedule = ChannelPruningSchedule(
                hparams.channel_ratio, hparams.channel_schedule,
                hparams.epochs, hparams.prune_warmup, hparams.channel_schedule_length, hparams.prune_cooldown, hparams.rewiring_freq
            )

            unpruned_layers = ["model.conv1", "model.conv_bn"]
            pruning_method = ChannelPruning(
                hparams.masking_type, pruning_schedule, importance_accum, coster,
                hparams.channel_type, unpruned_layers, hparams.channel_chunk_size, hparams.channel_allow_layer_prune, hparams.channel_bnrescaling_type,
                hparams.channel_doublesided_weight, track_mask_convergence=True
            )
        else:
            raise NotImplementedError("Pruning is set but an unrecognized configuration was given")

        pruning_callback = DynamicPruning(
            pruning_method, hparams.importance_type, hparams.importance_grad_type,
            hparams.importance_hess_type, hparams.pruned_decay, True,
            log_verbosity=PruningLogVerbosity.Full
        )
        callbacks.append(pruning_callback)

    plugins = []
    if accelerator == "ddp":
        plugins.append(DDPPlugin(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            ddp_comm_hook=fp16_compress_hook if hparams.fp16 else None
        ))

    trainer = pl.Trainer(
        accelerator=accelerator, num_nodes=hparams.num_nodes, gpus=hparams.gpus,
        benchmark=using_gpu, sync_batchnorm=sync_batchnorm,
        max_epochs=hparams.epochs, precision=precision, accumulate_grad_batches=accum_grad_batches,
        gradient_clip_val=hparams.clip, log_every_n_steps=hparams.train_log_freq,
        plugins=plugins, callbacks=callbacks, logger=logger, weights_summary="full"
    )

    # Run experiment
    trainer.fit(model, datamodule=dm)

    # Perform final validation
    trainer.validate(model, datamodule=dm)

    # Save the model (without training state)
    torch.save(model.model.state_dict(), f"{logger.log_dir}/last.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for testing pruning for image classifiction problems")

    # See https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md
    # for many of the hyperparameter choices

    # Architecture options
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50")
    parser.add_argument("--pretrained", type=str,
                        nargs="?", const=True, default=False,
                        help="whether to use pretrained network. If the optional string arg given, path to pretrained weights")

    # Dataset options
    parser.add_argument("--dataset", type=str, default="Imagenet",
                        help="The name of the dataset")
    parser.add_argument("--data-root", type=str, default="data",
                        help="The root directory of the dataset")
    parser.add_argument("--workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="The number of examples in one training batch")
    parser.add_argument("--simulated-batch-size", type=int, default=-1,
                        help="size of a total batch size, for simulating bigger batches")

    # Learning options
    parser.add_argument("--learning-rate", type=float, default=0.256,
                        help="learning rate for the optimizer (for batch size 256; increases linearly)")
    parser.add_argument("--momentum", default=0.875, type=float,
                        help="momentum")
    parser.add_argument("--nesterov", action="store_true",
                        help="use nesterov momentum, default: false)")
    parser.add_argument("--weight-decay", type=float, default=3.0517578125e-05,
                        help="weight decay factor for the optimizer")
    parser.add_argument("--bn-weight-decay", default=0.0, type=float,
                        help="weight decay on BN (default: 0.0)")
    parser.add_argument('--clip', default=None, type=float)

    parser.add_argument("--epochs", type=int, default=90,
                        help="Number of epochs to train")
    parser.add_argument("--lr-schedule", default="cosine", type=str, metavar="SCHEDULE",
                        choices=["step", "linear", "cosine", "step_prune"])
    parser.add_argument("--warmup", default=8, type=int,
                        metavar="E", help="number of warmup epochs")
    parser.add_argument("--label-smoothing", default=0.1, type=float,
                        metavar="S", help="label smoothing")

    # Trainer options
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed")
    parser.add_argument("--num-nodes", default=1, type=int,
                        help="Number of nodes to use (default: 1)")
    parser.add_argument("--gpus", default=None, type=int,
                        help="Number of GPUs to use (default: None)")
    parser.add_argument("--fp16", action="store_true",
                        help="Run model in fp16/AMP (automatic mixed precision) mode.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="The output directory to save the checkpoint and training log.")
    parser.add_argument("--train-log-freq", type=int, default=50,
                        help="The frequency (global step) to log the metrics during the training process")
    parser.add_argument("--ckpt-freq", type=int, default=10,
                        help="The frequency (epoch) to save the checkpoint")

    # Prune options
    parser.add_argument("--prune", action="store_true", default=False,
                        help="whether to prune")
    parser.add_argument("--masking-type", type=ParameterMaskingType, default=ParameterMaskingType.Soft,
                        action=EnumAction, help="type of masking")
    parser.add_argument("--prune-warmup", type=int, default=10, help="number of warmup epochs before starting to prune")
    parser.add_argument("--prune-cooldown", type=int, default=25, help="number of cooldown epochs to no longer prune")
    parser.add_argument("--rewiring-freq", type=int, default=80,
                        help="Number of batches of size 256 before rewiring")
    parser.add_argument("--pruned-decay", type=float, default=2e-4,
                        help="weight decay on pruned weights")

    parser.add_argument("--channel-type", type=ChannelPruningType, default=ChannelPruningType.Skip,
                        action=EnumAction, help="type of channel pruning")
    parser.add_argument("--channel-ratio", type=float, default=0, help="ratio of channels to prune")
    parser.add_argument("--channel-chunk-size", type=int, default=1, help="number of consecutive channels to prune together")
    parser.add_argument("--channel-allow-layer-prune", action="store_true", default=False,
                        help="whether to allow complete layers to be pruned by channel pruning")
    parser.add_argument("--channel-schedule", type=str, default="exp", help="type of pruning schedule")
    parser.add_argument("--channel-schedule-length", type=int, default=55, help="length of pruning schedule")
    parser.add_argument("--channel-bnrescaling-type", type=ChannelBNRescalingType, default=ChannelBNRescalingType.Skip,
                        action=EnumAction, help="type of BN rescaling to perform due to channel pruning")
    parser.add_argument("--channel-doublesided-weight", type=float, default=1,
                        help="fraction of cost that should be assigned to pruning input channels (as opposed to output layers of upstream layers)")

    parser.add_argument("--importance-type", type=ImportanceType, default=ImportanceType.Weight,
                        action=EnumAction, help="type of importance scoring")
    parser.add_argument("--importance-grad-type", type=ImportanceGradType, default=ImportanceGradType.INST,
                        action=EnumAction, help="type of gradient to use in importance scoring (for order >= 1)")
    parser.add_argument("--importance-hess-type", type=ImportanceHessType, default=ImportanceHessType.GRADSQ,
                        action=EnumAction, help="type of hessian to use in importance scoring (for order >= 2)")
    parser.add_argument("--importance-accumulator", type=ImportanceAccumulatorType, default=ImportanceAccumulatorType.Latest,
                        action=EnumAction, help="type of importance accumulation to use")

    parser.add_argument("--costing-type", type=CostingType, default=CostingType.Latency,
                        action=EnumAction, help="type of cost function")
    parser.add_argument("--costing-latency-table", type=str, default="./latency_tables/resnet50_titanV_cudnn74.pkl",
                        help="path to latency lookup table")

    # Parse the args
    args = parser.parse_args()
    if args.simulated_batch_size == -1:
        args.simulated_batch_size = args.batch_size
    elif args.simulated_batch_size % args.batch_size:
        raise argparse.ArgumentParser.error("Simulated batch size needs to be a multiple of the batch size")

    # Train classifier
    main(args)
