"""
A Tracker instance is responsible for:
1. tracking epochs, rounds, steps or any BaseTrackable instances.
2. tracking some curriculum- or annealing-related hyperparameters.
3. tracking metrics.
4. displaying a progress bar (through trackables.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial, update_wrapper
from typing import Any, Dict, List, Sequence, Type

from dev_misc.utils import deprecated

from .trackable import (BaseTrackable, MaxTrackable, MinTrackable,
                        TrackableRegistry, TrackableUpdater)

task_class: Type[dataclass] = update_wrapper(partial(dataclass, eq=False), dataclass)


@task_class
class Task:

    def __hash__(self):
        return id(self)

    def __eq__(self, other: Task):
        return id(self) == id(other)


class Tracker:

    def __init__(self):
        self.tasks: List[Task] = list()
        self.task_weights: List[Task] = list()

        self.trackable_reg = TrackableRegistry()

    def is_finished(self, name: str):
        return self.trackable_reg[name].value >= self.trackable_reg[name].total

    def add_trackable(self, name: str, *, total: int = None, endless: bool = False, agg_func: str = 'count') -> BaseTrackable:
        trackable = self.trackable_reg.register_trackable(name, total=total, endless=endless, agg_func=agg_func)
        return trackable

    def add_max_trackable(self, name: str) -> MaxTrackable:
        return self.add_trackable(name, agg_func='max')

    def add_min_trackable(self, name: str) -> MinTrackable:
        return self.add_trackable(name, agg_func='min')

    @deprecated
    def ready(self): ...

    def add_task(self, task: Task, weight: float):
        self.tasks.append(task)
        self.task_weights.append(weight)

    def add_tasks(self, tasks: Sequence[Task], weights: List[float]):
        if len(tasks) != len(weights):
            raise ValueError(f'Mismatched lengths from tasks ({len(tasks)}) and weights ({len(weights)}).')
        for task, weight in zip(tasks, weights):
            self.add_task(task, weight)

    def draw_task(self) -> Task:
        task = random.choices(self.tasks, weights=self.task_weights)[0]
        return task

    def __getattr__(self, attr: str):
        try:
            return self.trackable_reg[attr].value
        except KeyError:
            raise AttributeError(f'No trackable named {attr}.')

    def __getitem__(self, name: str):
        return self.trackable_reg[name]

    def update(self, name: str, *, value: Any = None) -> bool:
        """Update a trackable, and return whether it is updated."""
        trackable = self.trackable_reg[name]
        updater = TrackableUpdater(trackable)
        return updater.update(value=value)

    def reset(self, name: str):
        """Reset a trackable."""
        trackable = self.trackable_reg[name]
        trackable.reset()
