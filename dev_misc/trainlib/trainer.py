import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from dev_misc.utils import deprecated

from .metrics import Metrics
from .tracker.tracker import Tracker

...  # FIXME(j_luo) fill in this: move this to somewhere higher
def get_trainable_params(mod: torch.nn.Module, named: bool = True):
    if named:
        for name, param in mod.named_parameters():
            if param.requires_grad:
                yield name, param
    else:
        for param in mod.parameters():
            if param.requires_grad:
                yield param


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


def set_random_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def has_gpus() -> bool:
    return bool(os.environ.get('CUDA_VISIBLE_DEVICES', False))
