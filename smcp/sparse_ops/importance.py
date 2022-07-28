# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
from torch import fx, nn, optim


class ImportanceType(Enum):
    Weight = "Weight"
    TaylorFO = "TaylorFO"
    TaylorSO = "TaylorSO"

class ImportanceGradType(Enum):
    INST = "INST" # Instantaneous gradient from most recent batch
    OPT = "OPT" # Gradient just applied by the optimizer

class ImportanceHessType(Enum):
    GRADSQ = "GRADSQ" # Gradient outer product (TODO: Need motivation here)
    OPT = "OPT" # Use estimate from optimizer (Note: only implemented for certain optimizers)

class ImportanceFunction(ABC):
    def calculate(self) -> None:
        pass

    @abstractmethod
    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        """
        Returns the importance score of each parameter for remaining unpruned. Should seek to maximize the unpruned importance.
        """
        pass

    def cleanup(self) -> None:
        pass

class WeightImportance(ImportanceFunction):
    """"Zeroth" order. Aka weight importance (equivalent to all gradients being -sign(param))"""
    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        return param.abs()

class TaylorFOImportance(ImportanceFunction):
    """First order Taylor expansion term"""
    def __init__(self, grad_type: ImportanceGradType, optimizer: Optional[optim.Optimizer] = None):
        if grad_type == ImportanceGradType.OPT:
            assert optimizer is not None, f"Grad type {ImportanceGradType.OPT} requires an optimizer to be passed"

        self.grad_type = grad_type
        self.optimizer = optimizer

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        grad = self._get_gradient(param)
        assert grad is not None, "Cannot calculate TaylorFO importance without gradient"

        return - param * grad

    def _get_gradient(self, param: torch.Tensor) -> torch.Tensor:
        if self.grad_type == ImportanceGradType.INST:
            return param.grad
        elif self.grad_type == ImportanceGradType.OPT:
            if isinstance(self.optimizer, optim.Adam):
                state = self.optimizer.state[param]
                return state["exp_avg"]
            elif isinstance(self.optimizer, optim.SGD):
                state = self.optimizer.state[param]
                return state["momentum_buffer"]
            else:
                raise NotImplementedError(f"Grad type {self.grad_type} not supported for optimizer {type(self.optimizer)}")
        else:
            raise NotImplementedError(f"Grad type {self.grad_type} invalid")

class TaylorSOImportance(TaylorFOImportance):
    """Second order Taylor expansion terms, using diagonal Hessian"""
    def __init__(self, grad_type: ImportanceGradType, hess_type: ImportanceHessType, optimizer: Optional[optim.Optimizer] = None):
        super().__init__(grad_type, optimizer)

        self.hess_type = hess_type

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        grad = self._get_gradient(param)
        assert grad is not None, "Cannot calculate TaylorSO importance without gradient"

        diag_hess = self._get_diag_hess(param, grad)
        assert diag_hess is not None, "Cannot calculate TaylorSO importance without diag Hessian"

        return - param * grad + 0.5 * param.pow(2) * diag_hess

    # See https://arxiv.org/pdf/1612.01543.pdf Section 3.5
    # See https://arxiv.org/pdf/1502.04390.pdf Section 3
    def _get_diag_hess(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self.hess_type == ImportanceHessType.GRADSQ:
            return grad.pow(2)
        elif self.hess_type == ImportanceHessType.OPT:
            if isinstance(self.optimizer, optim.Adam):
                state = self.optimizer.state[param]
                return state["exp_avg_sq"]
            else:
                raise NotImplementedError(f"Hess type {self.hess_type} not supported for optimizer {type(self.optimizer)}")
        else:
            raise NotImplementedError(f"Hess type {self.hess_type} invalid")

def create_importance_function(
    importance_type: ImportanceType, model: nn.Module, optimizer: optim.Optimizer, grad_type: Optional[ImportanceGradType] = None, hess_type: Optional[ImportanceHessType] = None
) -> ImportanceFunction:
    if importance_type == ImportanceType.Weight:
        return WeightImportance()
    elif importance_type == ImportanceType.TaylorFO:
        assert grad_type is not None, f"Taylor first order importance requires the {ImportanceGradType} to be set"

        return TaylorFOImportance(grad_type, optimizer)
    elif importance_type == ImportanceType.TaylorFO:
        assert grad_type is not None, f"Taylor second order importance requires the {ImportanceGradType} to be set"
        assert hess_type is not None, f"Taylor second order importance requires the {ImportanceHessType} to be set"

        return TaylorSOImportance(grad_type, hess_type, optimizer)

    raise NotImplementedError(f"Importance type {importance_type} invalid")
