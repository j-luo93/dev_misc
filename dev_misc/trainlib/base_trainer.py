import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import tee
from typing import (Callable, Dict, List, NewType, Optional, Sequence, Tuple,
                    Union)

import torch
from torch.nn.init import xavier_normal_, xavier_uniform_

from .base_data_loader import BaseDataLoader, BaseDataLoaderRegistry
from .metrics import Metrics
from .tracker.tracker import Task, Tracker
from .trainer import get_trainable_params


@dataclass
class Callback:
    interval: int
    func: Callable


class BaseTrainer(ABC):
    # TODO(j_luo) Refactor this so that everything is a callback, and subclass this by implementing the basic workflow. Note that metric writers are observers.

    def __init__(self,
                 model,
                 tasks: Sequence[Task],
                 task_weights: Sequence[int],
                 main_tname: str,  # Main trackable name that's updated every step.
                 stage_tnames: Optional[Sequence[str]] = None,  # Names to compute the stage.
                 evaluator: Optional = None,
                 check_tname: str = 'check',
                 check_interval: Optional[int] = None,
                 eval_tname: str = 'eval',
                 eval_interval: Optional[int] = None,
                 save_tname: str = 'save',
                 save_interval: Optional[int] = None):
        self.tracker = Tracker()
        self.tracker.add_tasks(tasks, task_weights)
        self.add_trackables()
        self.model = model
        self.optimizer = None
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
        if self.eval_interval:
            self.tracker.add_trackable(eval_tname, total=self.eval_interval, endless=True)
            self.eval_tname = eval_tname
        if self.save_interval:
            self.tracker.add_trackable(save_tname, total=self.save_interval, endless=True)
            self.save_tname = save_tname
        self._callbacks: Dict[str, List[Callback]] = defaultdict(list)

    @abstractmethod
    def add_trackables(self, *args, **kwargs):
        """Add all trackables. `self.tracker` should be called here."""

    def add_callback(self, tname: str, interval: int, callback: Callable):
        # IDEA(j_luo) Rewrite check/eval/save functions using this method.
        if not tname in self.tracker:
            raise NameError(f'No trackable named {tname}.')

        self._callbacks[tname].append(Callback(interval, callback))

    def update(self, tname: str, metrics: Optional[Metrics] = None):
        """This does two things: call tracker and update a trackable, call registered callbacks."""
        self.tracker.update(tname)
        for callback in self._callbacks[tname]:
            if self.tracker[tname].value % callback.interval == 0:
                # FIXME(j_luo) This looks very hacky. Need a principled way of dealing with arguments for callbacks.
                if metrics is not None:
                    callback.func(metrics)
                else:
                    callback.func()

    def set_optimizer(self, optimizer_cls, **kwargs):
        params_cp0, params_cp1 = tee(get_trainable_params(self.model, named=False))
        self.optimizer = optimizer_cls(params_cp0, **kwargs)
        # Count number of params.
        total = sum([p.nelement() for p in params_cp1])
        logging.info(f'{total} trainable parameters will be optimized.')

    def set_lr_scheduler(self, scheduler_cls, **kwargs):
        if self.optimizer is None:
            raise RuntimeError(f'No optimizer has been set for lr_scheduler.')
        self.lr_scheduler = scheduler_cls(self.optimizer, **kwargs)

    def init_params(self, method='xavier_normal'):
        init_func = xavier_normal_ if method == 'xavier_normal_' else xavier_uniform_
        total = 0
        num_init = 0
        for param in get_trainable_params(self.model, named=False):
            if param.dim() == 2:
                init_func(param)
                num_init += param.numel()
            total += param.numel()
        logging.info(f'{num_init}/{total} trainable parameters initialized using {method}.')

    def train(self, dl_reg: BaseDataLoaderRegistry):
        metrics = Metrics()
        while not self.tracker.is_finished(*self.stage_tnames):
            task = self.tracker.draw_task()
            dl = dl_reg[task]
            step_metrics = self.train_one_step(dl)
            metrics += step_metrics

            self.update(self.main_tname)
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

        self.update(self.check_tname, metrics)
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
        logging.info(metrics.get_table(title=str(self.stage)))
        metrics.clear()

    def try_evaluate(self) -> Optional[Metrics]:
        if not self.eval_interval or self.evaluator is None:
            return

        self.update(self.eval_tname)
        if not self.tracker.is_finished(self.eval_tname):
            return

        self.model.eval()
        with torch.no_grad():
            return self.evaluate()

    def evaluate(self):
        eval_metrics = self.evaluator.evaluate(self.stage)
        logging.info(eval_metrics.get_table(title='Eval'))
        return eval_metrics

    def try_save(self, eval_metrics: Optional[Metrics]):
        if not self.save_interval:
            return

        self.update(self.save_tname)
        if not self.tracker.is_finished(self.save_tname):
            return

        self.save(eval_metrics)

    @abstractmethod
    def save(self, eval_metrics: Metrics): ...
