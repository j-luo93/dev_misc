import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import tee
from pathlib import Path
from typing import (Callable, Dict, List, NewType, Optional, Sequence, Tuple,
                    Type, Union)

import torch
import torch.nn as nn
from torch.nn.init import uniform_, xavier_normal_, xavier_uniform_

from .base_data_loader import BaseDataLoader, BaseDataLoaderRegistry
from .metrics import Metrics
from .tb_writer import MetricWriter
from .tracker.tracker import BaseSetting, Tracker
from .trainer import get_trainable_params

Optimizer = torch.optim.Optimizer


@dataclass
class Callback:
    name: str
    func: Callable


def init_params(model, method, *args, **kwargs) -> Tuple[int, int]:
    m2f = {
        'xavier_normal': xavier_normal_,
        'xavier_uniform': xavier_uniform_,
        'uniform': uniform_
    }
    init_func = m2f[method]
    total = 0
    num_init = 0
    for name, param in get_trainable_params(model, named=True):
        if param.dim() == 2:
            init_func(param, *args, **kwargs)
            num_init += param.numel()
            logging.info(f'Init: {name} {tuple(param.shape)}')
        else:
            logging.info(f'Skipped init: {name} {tuple(param.shape)}')
        total += param.numel()
    return num_init, total


class BaseTrainer(ABC):
    # TODO(j_luo) Refactor this so that everything is a callback, and subclass this by implementing the basic workflow. Note that metric writers are observers.
    # TODO(j_luo) Make this stateless?

    def __init__(self,
                 model,
                 settings: Sequence[BaseSetting],
                 setting_weights: Sequence[int],
                 main_tname: str,  # Main trackable name that's updated every step.
                 stage_tnames: Optional[Sequence[str]] = None,  # Names to compute the stage.
                 evaluator: Optional = None,
                 check_tname: str = 'check',
                 check_interval: Optional[int] = None,
                 eval_tname: str = 'eval',
                 eval_interval: Optional[int] = None,
                 save_tname: str = 'save',
                 save_interval: Optional[int] = None,
                 metric_writer: Optional[MetricWriter] = None):
        self.tracker = Tracker()
        self.tracker.add_settings(settings, setting_weights)
        self.add_trackables()
        self.model = model
        # Multiple optimizers are allowed.
        self.optimizers: Dict[str, Optimizer] = dict()
        self.lr_scheduler = None
        # IDEA(j_luo) Maybe we should just put all evaluation methods as part of trainer?
        self.evaluator = evaluator

        self.main_tname = main_tname
        self.stage_tnames = tuple(stage_tnames or [main_tname])

        self.check_interval = check_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval or eval_interval
        if self.check_interval:
            self.tracker.add_trackable(check_tname, total=self.check_interval, endless=True)
            self.check_tname = check_tname
        if self.eval_interval and evaluator is not None:
            self.tracker.add_trackable(eval_tname, total=self.eval_interval, endless=True)
            self.eval_tname = eval_tname
        if self.save_interval:
            self.tracker.add_trackable(save_tname, total=self.save_interval, endless=True)
            self.save_tname = save_tname
        self._callbacks: Dict[str, Dict[str, Callback]] = defaultdict(dict)

        self.metric_writer = metric_writer
        # `global_step` is an integer that can be used by the metric writer for writing to tensorboard. It is the number of steps (i.e., number of the batches) that has passed.
        # IDEA(j_luo) Make global_step and stage a global property?
        self._global_step = 0

    @property
    def optimizer(self) -> Optimizer:
        """Return the default optimizer."""
        if '__DEFAULT__' not in self.optimizers:
            raise AttributeError(f'No optimizer has been set.')
        return self.optimizers['__DEFAULT__']

    @abstractmethod
    def add_trackables(self, *args, **kwargs):
        """Add all trackables. `self.tracker` should be called here."""

    def add_callback(self, tname: str, cb_name: str, callback: Callable):
        # IDEA(j_luo) Rewrite check/eval/save functions using this method.
        if not tname in self.tracker:
            raise NameError(f'No trackable named {tname}.')

        # if cb_name not in self._callbacks[tname]:
        self._callbacks[tname][cb_name] = Callback(cb_name, callback)

    def _update(self, tname: str, metrics: Optional[Metrics] = None):
        """This does two things: call tracker and update a trackable, call registered callbacks."""
        self.tracker.update(tname)
        remaining = dict()
        for callback in self._callbacks[tname].values():
            if self.tracker.is_finished(tname):
                callback.func()
            else:
                remaining[callback.name] = callback
        self._callbacks[tname] = remaining

    def set_optimizer(self,
                      optimizer_cls: Type[Optimizer],
                      name: Optional[str] = None,
                      mod: Optional[nn.Module] = None, **kwargs):
        name = name or '__DEFAULT__'
        mod = mod or self.model
        params_cp0, params_cp1 = tee(get_trainable_params(mod, named=False))
        if name in self.optimizers:
            logging.warning(f'Resetting the optimizer named "{name}".')

        self.optimizers[name] = optimizer_cls(params_cp0, **kwargs)
        # Count number of params.
        total = sum([p.nelement() for p in params_cp1])
        logging.info(f'{total} trainable parameters will be optimized.')

    def set_lr_scheduler(self, scheduler_cls,
                         name: Optional[str] = None,
                         **kwargs):
        name = name or '__DEFAULT__'
        self.lr_scheduler = scheduler_cls(self.optimizers[name], **kwargs)

    def init_params(self, method, *args, **kwargs):
        num_init, total = init_params(self.model, method, *args, **kwargs)
        logging.imp(f'{num_init}/{total} trainable parameters initialized using {method}.')

    def train(self, dl_reg: BaseDataLoaderRegistry):
        metrics = Metrics()
        while not self.tracker.is_finished(*self.stage_tnames):
            setting = self.tracker.draw_setting()
            dl = dl_reg[setting]
            step_metrics = self.train_one_step(dl)
            metrics += step_metrics
            # FIXME(j_luo)  global step should be part of tracker.
            self._global_step += 1
            self._update(self.main_tname)

            self.try_check(metrics)
            eval_metrics = self.try_evaluate()
            self.try_save(eval_metrics)

            if self.should_terminate():
                break
        self.tracker.clear_trackables()

    def should_terminate(self) -> bool:
        """Return whether the training loop should be terminated or not."""
        return False

    @abstractmethod
    def train_one_step(self, dl: BaseDataLoader) -> Metrics:
        """Train one step."""

    def try_check(self, metrics: Metrics):
        if not self.check_interval:
            return

        self._update(self.check_tname, metrics)
        if not self.tracker.is_finished(self.check_tname):
            return

        self.check(metrics)

    @property
    def stage(self) -> str:
        """`stage` is a string representation of which training stage (i.e., step, round) the trainer is currrently in."""
        ret = list()
        for tname in self.stage_tnames:
            ret.append(str(self.tracker[tname].value))
        return '_'.join(ret)

    def check(self, metrics: Metrics):
        logging.info(metrics.get_table(title=str(self.stage), num_paddings=8))
        if self.metric_writer is not None:
            self.metric_writer.add_metrics(metrics, global_step=self._global_step)
            self.metric_writer.flush()
        metrics.clear()

    def try_evaluate(self) -> Optional[Metrics]:
        if not self.eval_interval or self.evaluator is None:
            return

        self._update(self.eval_tname)
        if not self.tracker.is_finished(self.eval_tname):
            return

        return self.evaluate()

    def evaluate(self):
        eval_metrics = self.evaluator.evaluate(self.stage, self._global_step)
        return eval_metrics

    def try_save(self, eval_metrics: Optional[Metrics]):
        if not self.save_interval:
            return

        self._update(self.save_tname)
        if not self.tracker.is_finished(self.save_tname):
            return

        self.save(eval_metrics)

    @abstractmethod
    def save(self, eval_metrics: Metrics): ...
