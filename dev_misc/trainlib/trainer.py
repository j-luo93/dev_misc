import os
import random
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from dev_misc.utils import deprecated

from .metrics import Metric, Metrics
from .tracker.tracker import Tracker


def get_trainable_params(mod: torch.nn.Module, named: bool = True):
    # FIXME(j_luo) fill in this: move this to somewhere higher
    if named:
        for name, param in mod.named_parameters():
            if param.requires_grad:
                yield name, param
    else:
        for param in mod.parameters():
            if param.requires_grad:
                yield param


def clip_grad(params: Iterator[nn.Parameter], batch_size: int, max_norm: float = 5.0) -> Metric:
    grad_norm = clip_grad_norm_(params, max_norm)
    grad_norm = Metric('grad_norm', grad_norm, batch_size)
    return grad_norm


def freeze(mod: nn.Module):
    """Freeze all parameters within a module."""
    for p in mod.parameters():
        p.requires_grad = False
    for m in mod.children():
        freeze(m)


def get_grad_norm(mod: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def get_optim_params(optim: torch.optim.Optimizer) -> Iterator[nn.Parameter]:
    for param_group in optim.param_groups:
        yield from param_group['params']


def set_random_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def has_gpus() -> bool:
    return bool(os.environ.get('CUDA_VISIBLE_DEVICES', False))
