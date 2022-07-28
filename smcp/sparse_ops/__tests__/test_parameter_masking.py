# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from copy import deepcopy
from typing import List

import pytest
import torch
from torch import nn

from smcp.sparse_ops.parameter_masking import add_masking, has_mask, get_mask, set_mask, get_orig, remove_masking, ParameterMaskingType

def _test_parameter_masking(
    clean_model: nn.Module, input_shape: torch.Size, masking_type: ParameterMaskingType
) -> List[str]:
    """
    Test the masked model behaves as if the weights were stored as 0s in the clean model.

    Args:
    -----
        - clean_model: baseline model
        - input_shape: shape of input the models expect
        - masking_type: type of parameter masking

    Return:
    -------
    List of parameters checked
    """

    # Create version of model with parameter masking
    model = deepcopy(clean_model)
    add_masking(model, masking_type)

    # Make sure the models share parameters and apply same random mask to both models
    for name, clean_orig in clean_model.named_parameters():
        if has_mask(model, name):
            mask = torch.randint(0, 2, clean_orig.shape, dtype=torch.bool)
            set_mask(model, name, mask)
            setattr(clean_model, name, torch.nn.Parameter((clean_orig * mask).detach().clone()))

    # Compare to a simple model with an equivalent weight matrix
    input_c = torch.randn(1, *input_shape, requires_grad=True)
    input_c.retain_grad()

    output_c = clean_model(input_c)
    output_c.retain_grad()

    target = torch.randn_like(output_c)
    loss_c = torch.norm(output_c - target)
    loss_c.backward()

    # Want forward pass to use sparse weights but want backward pass to update all weights
    input = input_c.detach().clone()
    input.requires_grad = True
    input.retain_grad()

    output = model(input)
    output.retain_grad()

    loss = torch.norm(output - target)
    loss.backward()

    # Check the gradients are the same
    assert torch.allclose(input.grad, input_c.grad), "Input gradients don't match"
    for name, clean_orig in clean_model.named_parameters():
        if has_mask(model, name):
            model_param = get_orig(model, name)
            model_mask = get_mask(model, name)

            assert model_mask.grad == None, f"Gradients exist on the mask for {name}"

            if masking_type == ParameterMaskingType.Soft:
                assert torch.allclose(clean_orig.grad, model_param.grad), f"Parameter gradient {name} doesn't match"

                # (Double) Check the gradients still flow to masked weights
                assert torch.count_nonzero(model_param.grad * ~model_mask) > 0, f"No gradients flowing to masked part of param {name}"
            elif masking_type == ParameterMaskingType.Hard or masking_type == ParameterMaskingType.Permanent:
                assert torch.allclose(clean_orig.grad * model_mask, model_param.grad), f"Parameter gradient {name} doesn't match"

                # (Double) Check the gradients do not flow to masked weights
                assert torch.count_nonzero(model_param.grad * ~model_mask) == 0, f"Gradients flowing to masked part of param {name}"
            else:
                raise NotImplementedError(f"{masking_type} not recognized")
        else:
            assert torch.allclose(clean_orig.grad, getattr(model, name).grad), f"Parameter gradient {name} doesn't match"

    checked_params = [name for name, _ in clean_model.named_parameters() if has_mask(model, name)]

    # Make the sparsity permanent
    remove_masking(model)
    for name, clean_orig in clean_model.named_parameters():
        model_param = getattr(model, name)
        assert torch.allclose(clean_orig, model_param), f"Parameter {name} doesn't match after mask removed"
    for name, _ in model.named_buffers():
        assert not name.endswith("_mask"), f"Mask on {name} not removed"
    for name, _ in model.named_parameters():
        assert not name.endswith("_orig"), f"Original/dense parameter on {name} not removed"

    return checked_params

masking_types = [ParameterMaskingType.Soft, ParameterMaskingType.Hard, ParameterMaskingType.Permanent]

@pytest.mark.parametrize("masking_type", masking_types)
def test_linear_masking(masking_type: ParameterMaskingType) -> None:
    clean_model = nn.Linear(10, 5, bias=False)

    tested_params = _test_parameter_masking(clean_model, (10,), masking_type)

    assert tested_params == ["weight"]

@pytest.mark.parametrize("masking_type", masking_types)
def test_conv2d_masking(masking_type: ParameterMaskingType) -> None:
    clean_model = nn.Conv2d(10, 5, 3, bias=False)

    tested_params = _test_parameter_masking(clean_model, (10, 32, 32), masking_type)

    assert tested_params == ["weight"]
