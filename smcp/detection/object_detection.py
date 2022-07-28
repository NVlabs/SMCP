# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
from math import ceil
from typing import Any, Dict, Optional, Tuple
import warnings

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
import torch
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import torch.nn as nn
import torch.optim as optim

from smcp.core.enum_parse import EnumAction
from smcp.core.lr_scheduler import WarmupCustomMultiStepLR
from smcp.detection.datasets.PascalVOC import PascalVOCDataModule, PascalVOCDataBatch, get_VOC_label_map
from smcp.detection.losses import MultiBoxLoss
from smcp.detection.metrics import SSDDetectionMAP
from smcp.detection.models.ssd import get_ssd_model, create_VOC_prior_boxes
from smcp.sparse_ops import create_coster, create_importance_accumulator, ChannelBNRescalingType, ChannelPruning, ChannelPruningSchedule, ChannelPruningType, CostingType, DynamicPruning, ImportanceAccumulatorType, ImportanceType, ImportanceGradType, ImportanceHessType, ParameterMaskingType, PruningLogVerbosity

# Disable pl deprecations
warnings.simplefilter("ignore", LightningDeprecationWarning)

class ObjectDetectorParams:
    arch: str
    backbone: str
    num_classes: int
    pretrained: Optional[str]
    learning_rate: float
    momentum: float
    nesterov: bool
    weight_decay: float
    bn_weight_decay: float
    epochs: int

class ObjectDetector(pl.LightningModule):
    hparams: ObjectDetectorParams
    model: nn.Module

    def __init__(
        self, arch: str, backbone: str, num_classes: int, pretrained: Optional[str],
        learning_rate: float, momentum: float, nesterov: bool, weight_decay: float, bn_weight_decay: float,
        epochs: int, **kwargs
    ):
        """Object Detector model

        Args:
            arch: type of detector architecture
            backbone: type of detector backbone
            num_classes: number of image classes
            pretrained: whether to use a pretrained network. If a string, path to the pretrained weights
            learning_rate: learning rate
            momentum: SGD momentum
            nesterov: whether to enable Nesterov momentum
            weight_decay: amount of weight decay for non-BatchNorm weights
            bn_weight_decay: amount of weight decay for BatchNorm weights
            epochs: total number of training epochs
        """
        super().__init__()
        self.save_hyperparameters()

        is_ssd300 = (arch == "SSD300")

        self.model = get_ssd_model(arch, backbone, num_classes=num_classes, pretrained=pretrained, batch_norm=True).to(memory_format=torch.channels_last)
        if is_ssd300:
            self.example_input_array = torch.ones(1, 3, 300, 300).to(memory_format=torch.channels_last)
        else:
            self.example_input_array = torch.ones(1, 3, 512, 512).to(memory_format=torch.channels_last)

        # Prior boxes
        priors_cxcy = create_VOC_prior_boxes(is_ssd300)
        self.criterion = MultiBoxLoss(priors_cxcy=priors_cxcy)

        label_map = get_VOC_label_map()
        self.test_map_metric = SSDDetectionMAP(label_map, priors_cxcy)

    def configure_optimizers(self) -> optim.Optimizer:
        parameters_for_optimizer = list(self.model.named_parameters())

        bn_biases = [
            v for n, v in parameters_for_optimizer
            if ("bn" in n) and (n.endswith(".bias"))
        ]
        bn_params = [
            v for n, v in parameters_for_optimizer
            if ("bn" in n) and not (n.endswith(".bias"))
        ]
        biases = [
            v for n, v in parameters_for_optimizer
            if not ("bn" in n) and (n.endswith(".bias"))
        ]
        rest = [
            v for n, v in parameters_for_optimizer
            if not ("bn" in n) and not (n.endswith(".bias"))
        ]

        optimizer = torch.optim.SGD(
            params=[
                {'params': biases, 'lr': 2 * self.hparams.learning_rate},
                {'params': rest},
                {'params': bn_biases, 'lr': 2 * self.hparams.learning_rate, 'weight_decay': self.hparams.bn_weight_decay},
                {'params': bn_params, 'weight_decay': self.hparams.bn_weight_decay}
            ],
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov
        )

        # Warmup to lr linearly over first 50
        # constant lr until 600
        # constant 3/8 * lr until 700
        # constant 1/8 * lr until 740
        # constant 1/20 * lr until 770
        # constant 1/200 * lr until 800
        warmup_length = 50
        milestones = [600, 700, 740, 770]
        gammas = [3/8, 1/3, 2/5, 1/10]
        lr_scheduler = WarmupCustomMultiStepLR(optimizer, warmup_length, milestones, gammas)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: optim.Optimizer, optimizer_idx: int) -> None:
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: PascalVOCDataBatch, batch_idx: int) -> torch.Tensor:
        images, boxes, labels, _ = batch

        predicted_locs, predicted_logits = self.forward(images) # (N, *, 4), (N, *, n_classes)
        preds = (predicted_locs, predicted_logits)
        target = (boxes, labels)

        loss = self.criterion(preds, target)
        self.log("train/loss", loss, sync_dist=True, batch_size=images.shape[0])

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images, boxes, labels, _ = batch

        predicted_locs, predicted_logits = self.forward(images) # (N, *, 4), (N, *, n_classes)

        preds = (predicted_locs, predicted_logits)
        target = (boxes, labels)
        loss = self.criterion(preds, target)

        self.log("val/loss", loss, sync_dist=True, batch_size=images.shape[0])

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images, boxes, labels, difficulties = batch

        predicted_locs, predicted_logits = self.forward(images) # (N, *, 4), (N, *, n_classes)

        preds = (predicted_locs, predicted_logits)
        target = (boxes, labels)
        loss = self.criterion(preds, target)

        target = (boxes, labels, difficulties)
        self.test_map_metric(preds, target)

        self.log("test/loss", loss, sync_dist=True, batch_size=images.shape[0])

        return loss

    def test_epoch_end(self, _: Any) -> None:
        acc_metrics = self.test_map_metric.compute()
        self.log_dict({ f"test/{k}": v for k,v in acc_metrics.items() })

def main(hparams):
    # Interpret/modify the hparams
    using_gpu = hparams.gpus is not None
    data_dtype = torch.float16 if hparams.fp16 else torch.float32
    precision = 16 if hparams.fp16 else 32
    accum_grad_batches = hparams.simulated_batch_size // hparams.batch_size
    accelerator = "ddp" if using_gpu else None
    sync_batchnorm =  using_gpu and hparams.batch_size <= 32

    eff_batch_size = hparams.batch_size * accum_grad_batches * hparams.num_nodes * (hparams.gpus if using_gpu else 1)
    hparams.learning_rate *= eff_batch_size / 128
    hparams.rewiring_freq = ceil(128 * hparams.rewiring_freq / eff_batch_size)

    if hparams.arch == "SSD300":
        dims = (300, 300)
    else:
        dims = (512, 512)

    # Setup datamodule
    if hparams.dataset != "PascalVOC":
        raise NotImplementedError(f"Dataset {hparams.dataset} unknown")

    dm = PascalVOCDataModule(
        hparams.data_root, dims=dims, keep_difficult=(not hparams.disable_difficult),
        num_workers=hparams.workers, batch_size=hparams.batch_size,
        shuffle=False, pin_memory=using_gpu, drop_last=True, dtype=data_dtype
    )

    # Setup model
    model = ObjectDetector(num_classes=dm.num_classes, **vars(hparams))

    # Setup trainer
    pl.seed_everything(hparams.seed, workers=True)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=hparams.output_dir,
        name=f"object_detector-{hparams.dataset}",
        default_hp_metric=False
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(
            filename="object_detector-epoch{epoch}-val_loss{val/loss:.4f}",
            mode="min",
            monitor="val/loss",
            auto_insert_metric_name=False,
            save_last=True,
            every_n_val_epochs=hparams.ckpt_freq
        )
    ]

    if hparams.prune:
        importance_accum = create_importance_accumulator(hparams.importance_accumulator)

        if hparams.channel_type is not ChannelPruningType.Skip:
            coster = create_coster(hparams.costing_type, hparams.costing_latency_table, hparams.ignore_lut_warnings)

            pruning_schedule = ChannelPruningSchedule(
                hparams.channel_ratio, hparams.channel_schedule,
                hparams.epochs, hparams.prune_warmup, hparams.channel_schedule_length, hparams.prune_cooldown, hparams.rewiring_freq
            )

            unpruned_layers = ["model.f_0.conv1"]
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
        log_every_n_steps=hparams.train_log_freq, check_val_every_n_epoch=hparams.ckpt_freq,
        plugins=plugins, callbacks=callbacks, logger=logger, weights_summary="full"
    )

    # Run experiment
    trainer.fit(model, datamodule=dm)

    # Run test
    trainer.test(model, datamodule=dm)

    # Save the model (without training state)
    torch.save(model.model.state_dict(), f"{logger.log_dir}/last.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for testing pruning for object detection problems")

    # Architecture options
    parser.add_argument("-a", "--arch", metavar="ARCH", default="SSD512", choices=["SSD512", "SSD300"])
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="optional path to pretrained network weights")

    # Dataset options
    parser.add_argument("--dataset", type=str, default="PascalVOC",
                        help="The name of the dataset")
    parser.add_argument("--data-root", type=str, default="data",
                        help="The root directory of the dataset")
    parser.add_argument("--workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="The number of examples in one training batch (per gpu)")
    parser.add_argument("--simulated-batch-size", type=int, default=-1,
                        help="size of a total batch size, for simulating bigger batches")
    parser.add_argument('--disable-difficult', action='store_true')

    # Learning options
    parser.add_argument("--learning-rate", type=float, default=8e-3,
                        help="learning rate for the optimizer (for batch size 128; increases linearly)")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="momentum")
    parser.add_argument("--nesterov", action="store_true",
                        help="use nesterov momentum, default: false)")
    parser.add_argument("--weight-decay", type=float, default=2e-3,
                        help="weight decay factor for the optimizer")
    parser.add_argument("--bn-weight-decay", default=0.0, type=float,
                        help="weight decay on BN (default: 0.0)")

    parser.add_argument("--epochs", type=int, default=800,
                        help="Number of epochs to train")

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
    parser.add_argument("--ckpt-freq", type=int, default=20,
                        help="The frequency (epoch) to save the checkpoint")

    # Prune options
    parser.add_argument("--prune", action="store_true", default=False,
                        help="whether to do pruning")
    parser.add_argument("--masking-type", type=ParameterMaskingType, default=ParameterMaskingType.Soft,
                        action=EnumAction, help="type of masking")
    parser.add_argument("--prune-warmup", type=int, default=60, help="number of warmup epochs before starting to prune")
    parser.add_argument("--prune-cooldown", type=int, default=350, help="number of cooldown epochs to no longer prune")
    parser.add_argument("--rewiring-freq", type=int, default=80,
                        help="Number of batches of size 128 before rewiring")
    parser.add_argument("--pruned-decay", type=float, default=2e-4,
                        help="weight decay on pruned weights")

    parser.add_argument("--channel-type", type=ChannelPruningType, default=ChannelPruningType.Skip,
                        action=EnumAction, help="type of channel pruning")
    parser.add_argument("--channel-ratio", type=float, default=0, help="ratio of channels to prune")
    parser.add_argument("--channel-chunk-size", type=int, default=1, help="number of consecutive channels to prune together")
    parser.add_argument("--channel-allow-layer-prune", action="store_true", default=False,
                        help="whether to allow complete layers to be pruned by channel pruning")
    parser.add_argument("--channel-schedule", type=str, default="exp", help="type of pruning schedule")
    parser.add_argument("--channel-schedule-length", type=int, default=250, help="length of pruning schedule")
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
    parser.add_argument("--ignore-lut-warnings", action="store_true", help="whether to ignore LUT warnings")

    # Parse the args
    args = parser.parse_args()
    if args.simulated_batch_size == -1:
        args.simulated_batch_size = args.batch_size
    elif args.simulated_batch_size % args.batch_size:
        raise argparse.ArgumentParser.error("Simulated batch size needs to be a multiple of the batch size")

    # Train classifier
    main(args)
