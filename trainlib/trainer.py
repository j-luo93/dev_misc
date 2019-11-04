import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from .metrics import Metrics
from .tracker import Tracker


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


class Trainer(ABC):
    """A base Trainer class that defines the basic workflow of a trainer."""

    def __init__(self):
        self.tracker = Tracker()

    @abstractmethod
    def check_metrics(self, accum_metrics: Metrics):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def train_loop(self, *args, **kwargs) -> Metrics:
        pass

    def train(self, *args, **kwargs):
        accum_metrics = Metrics()
        while not self.tracker.is_finished:
            metrics = self.train_loop(*args, **kwargs)
            accum_metrics += metrics
            self.tracker.update()

            self.check_metrics(accum_metrics)
            self.save()
