# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Any, Callable, Tuple

import torch
from torch import autograd, nn

# TODO: Consolidate with parameter_masking (lots of duplicate code)

class BNScaler(autograd.Function):
    # Want the gradients to work as if we had applied the scaler directly
    @staticmethod
    def forward(ctx, weight: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scaling)
        return weight * scaling

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        scaling, = ctx.saved_tensors

        new_grad_output = torch.zeros_like(grad_output)
        new_grad_output[scaling != 0] = grad_output[scaling != 0] / scaling[scaling != 0]

        return new_grad_output, None

def get_bn_scaling(module: nn.Module) -> torch.Tensor:
    return getattr(module, "weight_scaling")

def set_bn_scaling(module: nn.Module, new_scaling: torch.Tensor) -> None:
    module.weight_scaling[:] = new_scaling

def _create_bn_scaling_pre_hook() -> Callable[[nn.Module, Any], None]:
    def _scaling_pre_hook(module: nn.Module, _: Any) -> None:
        assert hasattr(module, "weight_orig"), f"Module {module} weight wasn't setup to be scaled"

        scaling = get_bn_scaling(module)
        orig = getattr(module, "weight_orig")

        setattr(module, "weight", BNScaler.apply(orig, scaling))

    return _scaling_pre_hook

def _create_bn_scaling_cleanup_hook() -> Callable[[nn.Module, Any], None]:
    def _scaling_cleanup_hook(module: nn.Module, input: Any, output: Any) -> None:
        assert hasattr(module, "weight_orig"), f"Module {module} weight wasn't setup to be pruned"

        setattr(module, "weight", None)

    return _scaling_cleanup_hook

def _add_bn_weight_scaling(module: nn.modules.batchnorm._BatchNorm) -> None:
    assert module.affine, "Cannot apply weight scaling to non-affine BN"

    weight = module.weight

    if weight is None or "weight" not in module._parameters.keys():
        return

    # Move original weight to new name
    module.register_parameter(f"weight_orig", weight)
    del module._parameters["weight"]

    # Register the scaling buffer
    scaling = torch.ones_like(weight)
    module.register_buffer(f"weight_scaling", scaling)

    # Register forward pre hook that will recompute the plain attribute before each forward
    remove_prehook = module.register_forward_pre_hook(_create_bn_scaling_pre_hook())
    setattr(module, f"weight_remove_prehook", remove_prehook)

    # Register forward hook that will cleanup the plain attribute after each forward
    # Note: without this, the model cannot be deepcopied (due to non-leaf tensors)
    remove_posthook = module.register_forward_hook(_create_bn_scaling_cleanup_hook())
    setattr(module, f"weight_remove_posthook", remove_posthook)

def add_bn_weight_scaling(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        _add_bn_weight_scaling(module)

def _remove_bn_weight_scaling(module: nn.modules.batchnorm._BatchNorm) -> None:
    is_parameter = "weight_orig" in module._parameters.keys()

    if is_parameter:
        orig = module._parameters["weight_orig"]
    else:
        return

    # update the orig with the current scaling
    scaling = module._buffers["weight_scaling"]
    orig.data = (orig * scaling).detach()

    # delete
    remove_prehook = getattr(module, "weight_remove_prehook")
    remove_prehook.remove()
    delattr(module, "weight_remove_prehook")

    remove_posthook = getattr(module, "weight_remove_posthook")
    remove_posthook.remove()
    delattr(module, "weight_remove_posthook")

    del module._buffers["weight_scaling"]
    del module._parameters["weight_orig"]

    setattr(module, "weight", orig)

def remove_bn_weight_scaling(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        _remove_bn_weight_scaling(module)
