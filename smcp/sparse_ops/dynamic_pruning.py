# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# Inspired by https://github.com/PyTorchLightning/pytorch-lightning/blob/580a3b5e3226671d60b619f4f2c44499fdda0cfa/pytorch_lightning/callbacks/pruning.py

from copy import deepcopy
from enum import Enum
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_debug
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.optim import Optimizer

from smcp.sparse_ops.base_pruning import BasePruningMethod
from smcp.sparse_ops.decay import apply_pruned_decay
from smcp.sparse_ops.importance import create_importance_function, ImportanceType, ImportanceGradType, ImportanceHessType
from smcp.sparse_ops.result import PruneResult

class PruningLogVerbosity(Enum):
    Simple = "Simple"
    Full = "Full"

class DynamicPruning(Callback):
    def __init__(
        self,
        pruning_method: BasePruningMethod,
        importance_type: ImportanceType, importance_grad_type: ImportanceGradType, importance_hess_type: ImportanceHessType,
        pruned_decay: float = 2e-4, make_pruning_permanent: bool = True,
        log_verbosity: PruningLogVerbosity = PruningLogVerbosity.Simple
    ) -> None:
        """
        Model pruning callback, using custom dynamic pruning code.
        This callback is responsible of pruning networks parameters during training.

        Args:
            pruning_method: pruning method to apply
            importance_type: importance scoring type
            importance_grad_type: type of gradient to use in importance scoring (for order >= 1)
            importance_hess_type: type of hessian to use in importance scoring (for order >= 2)
            pruned_decay: amount of weight decay for pruned weights
            make_pruning_permanent: Whether to remove all reparametrization pre-hooks and apply masks
                when training ends or the model is saved.
            log_verbosity: level of pruning logging to perform
        """
        self.pruning_method = pruning_method
        self.importance_type = importance_type
        self.importance_grad_type = importance_grad_type
        self.importance_hess_type = importance_hess_type
        self.pruned_decay = pruned_decay
        self.make_pruning_permanent = make_pruning_permanent
        self.log_verbosity = log_verbosity

        self._importance_fn = None

    def _log_prune_parameters(self, logger: LightningLoggerBase) -> None:
        logger.log_hyperparams({
            "importance_type": self.importance_type,
            "importance_grad_type": self.importance_grad_type,
            "importance_hess_type": self.importance_hess_type,
            "pruned_decay": self.pruned_decay,
            "make_pruning_permanent": self.make_pruning_permanent
        })

    # Setup masking after DDP  any optimizers or DDP
    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.training:
            self.pruning_method.apply_masking(pl_module)

    @torch.no_grad()
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_prune_parameters(trainer.logger)

        self._importance_fn = create_importance_function(
            self.importance_type, pl_module, trainer.optimizers[0],
            self.importance_grad_type, self.importance_hess_type
        )

        if self.pruning_method.should_update_importance(trainer.current_epoch, trainer.global_step):
            self._importance_fn.calculate()
            self.pruning_method.update_importance(pl_module, self._importance_fn)
            self._importance_fn.cleanup()

        if self.pruning_method.should_prune(trainer.current_epoch, trainer.global_step):
            rank_zero_debug("`ModelPruning.on_train_start`. Pruning before the first step.")

            self.pruning_method.prune(pl_module, trainer.current_epoch, trainer.global_step)
            self.pruning_method.reset_importance()

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer, opt_idx: int) -> None:
        pl_module.apply(lambda mod: apply_pruned_decay(mod, self.pruned_decay))

    @torch.no_grad()
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if not all((p.grad is not None) and torch.isfinite(p.grad).all() for p in pl_module.parameters()):
            # Do not prune or update importance for None/NaN/Inf gradients (None were deliberately skipped; rest likely due to AMP and will resolve on its own)
            return

        if self.pruning_method.should_update_importance(trainer.current_epoch, trainer.global_step):
            self._importance_fn.calculate()
            self.pruning_method.update_importance(pl_module, self._importance_fn)
            self._importance_fn.cleanup()

        if (trainer.global_step > 0) and self.pruning_method.should_prune(trainer.current_epoch, trainer.global_step):
            rank_zero_debug(f"`ModelPruning.on_train_batch_end`. Pruning at global step {trainer.global_step}")

            result = self.pruning_method.prune(pl_module, trainer.current_epoch, trainer.global_step)
            self.pruning_method.reset_importance()

            self._log_prune_result(trainer, result)

    @torch.no_grad()
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Log information about mask convergence per layer
        mask_conv, layer_conv_dict = self.pruning_method.get_mask_convergence()

        for step, count in zip(mask_conv[0], mask_conv[1]):
            trainer.logger.log_metrics({
                f"prune/mask_conv": count,
            }, step=step)

        if self.log_verbosity == PruningLogVerbosity.Full:
            for layer_name, (steps, counts) in layer_conv_dict.items():
                for step, count in zip(steps, counts):
                    trainer.logger.log_metrics({
                        f"prune/{layer_name}/mask_conv": count,
                    }, step=step)

        if self.make_pruning_permanent:
            rank_zero_debug("`ModelPruning.on_train_end`. Pruning is made permanent for this checkpoint")
            self.pruning_method.remove_masking(pl_module)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        # TODO: Blocked by the issue in https://github.com/PyTorchLightning/pytorch-lightning/issues/7505
        # if self.make_pruning_permanent:
        #     rank_zero_debug("`ModelPruning.on_save_checkpoint`. Pruning is made permanent for this checkpoint")
        #     # save a copy so training can continue with the same buffers
        #     copy = deepcopy(pl_module).to("cpu")
        #     self.pruning_method.remove_masking(copy)
        #     checkpoint["state_dict"] = copy.state_dict()

        return checkpoint

    def _log_prune_result(self, trainer: pl.Trainer, result: PruneResult) -> None:
        prune_metrics = result.todict("prune/", self.log_verbosity == PruningLogVerbosity.Full)

        trainer.logger.log_metrics(prune_metrics, step=trainer.global_step)
