import logging
from abc import ABC, abstractmethod
from itertools import tee
from typing import NewType, Optional, Sequence

from .base_data_loader import BaseDataLoader, BaseDataLoaderRegistry
from .metrics import Metrics
from .tracker.tracker import Task, Tracker
from .trainer import get_trainable_params


class BaseTrainer(ABC):

    def __init__(self,
                 model,
                 tasks: Sequence[Task],
                 task_weights: Sequence[int],
                 main_tname: str,
                 evaluator: Optional = None,
                 check_tname: str = 'check',
                 check_interval: Optional[int] = None,
                 eval_tname: str = 'eval',
                 eval_interval: Optional[int] = None):
        self.tracker = Tracker()
        self.tracker.add_tasks(tasks, task_weights)
        self.add_trackables()
        self.model = model
        self.optimizer = None
        # IDEA(j_luo) Maybe we should just put all evaluation methods as part of trainer?
        self.evaluator = evaluator

        self.main_tname = main_tname

        self.check_tname = check_tname
        self.eval_tname = eval_tname
        self.check_interval = check_interval
        self.eval_interval = eval_interval
        if check_interval:
            self.tracker.add_trackable(check_tname, total=check_interval, endless=True)
        if eval_interval:
            self.tracker.add_trackable(eval_tname, total=eval_interval, endless=True)

    @abstractmethod
    def add_trackables(self, *args, **kwargs):
        """Add all trackables. `self.tracker` should be called here."""

    def set_optimizer(self, optimizer_cls, **kwargs):
        params_cp0, params_cp1 = tee(get_trainable_params(self.model, named=False))
        self.optimizer = optimizer_cls(params_cp0, **kwargs)
        # Count number of params.
        total = sum([p.nelement() for p in params_cp1])
        logging.info(f'Found {total} trainable parameters.')

    def train(self, dl_reg: BaseDataLoaderRegistry):
        metrics = Metrics()
        while not self.tracker.is_finished(self.main_tname):
            task = self.tracker.draw_task()
            dl = dl_reg[task]
            step_metrics = self.train_one_step(dl)
            metrics += step_metrics

            self.tracker.update(self.main_tname)
            self.try_check(metrics)
            eval_metrics = self.try_evaluate()
            self.try_save(eval_metrics)

    @abstractmethod
    def train_one_step(self, dl: BaseDataLoader) -> Metrics:
        """Train one step."""

    def try_check(self, metrics: Metrics):
        if not self.check_interval:
            return

        self.tracker.update(self.check_tname)
        if not self.tracker.is_finished(self.check_tname):
            return

        self.check(metrics)

    def check(self, metrics: Metrics):
        logging.info(metrics.get_table(title=str(self.tracker[self.main_tname])))
        metrics.clear()

    def try_evaluate(self) -> Optional[Metrics]:
        if not self.eval_interval or self.evaluator is None:
            return

        self.tracker.update(self.eval_tname)
        if not self.tracker.is_finished(self.eval_tname):
            return

        return self.evaluate()

    def evaluate(self):
        eval_metrics = self.evaluator.evaluate(self.tracker)
        logging.info(eval_metrics.get_table(title='Eval'))
        return eval_metrics

    def try_save(self, eval_metrics: Optional[Metrics]):
        if not eval_metrics:
            return

        self.save(eval_metrics)

    @abstractmethod
    def save(self, eval_metrics: Metrics): ...
