from __future__ import annotations

from typing import Optional
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, overload

from dev_misc.utils import manager


class PBarOutOfBound(Exception):
    pass


class BaseTrackable(ABC):
    # IDEA(j_luo) Trackable and Tracker classes can add_trackable and put it in a registry. Maybe we can abstract some class out of it.
    # IDEA(j_luo) Maybe merge sth with Metric?

    def __init__(self, name: str, *, parent: Optional[BaseTrackable] = None, registry: Optional[TrackableRegistry] = None):
        self._name = name

        self.children: List[BaseTrackable] = list()
        if parent is not None:
            # NOTE(j_luo) Every child here would be reset after the parent is updated.
            parent.children.append(self)

        self.registry = registry if registry is not None else TrackableRegistry()

        self.reset()

    def __str__(self):
        return f'{self.name}: {self.value}'

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def value(self):
        """Value of this object."""

    @abstractmethod
    def reset(self):
        """Reset this object. Note that this is called inside __init__."""

    @abstractmethod
    def update(self) -> bool:
        """Update this object and return whether the value is updated."""

    def add_trackable(self, name: str, *, total: int = None) -> BaseTrackable:
        trackable = self.registry.register_trackable(name, total=total, parent=self)
        return trackable


class CountTrackable(BaseTrackable):

    def __init__(self, name: str, total: int, endless: bool = False, **kwargs):
        self._total = total
        self._endless = endless
        self._pbar = manager.counter(desc=name, total=total)
        super().__init__(name, **kwargs)

    @property
    def total(self):
        return self._total

    def update(self) -> bool:
        if self.value == self.total and self._endless:
            self.reset()

        while True:
            try:
                self._pbar.update()
                break
            except RuntimeError:
                logging.exception('Encountered some issue when updating the progress bar. Waiting to retry again')
                time.sleep(1)

        if self._total is not None and self._pbar.count > self._total:
            raise PBarOutOfBound(f'Progress bar ran out of bound.')
        for trackable in self.children:
            trackable.reset()
        return True

    def reset(self):
        self._pbar.start = time.time()
        self._pbar.count = 0
        self._pbar.refresh()

    @property
    def value(self):
        return self._pbar.count

    @property
    def is_finished(self) -> bool:
        return self._total and self._pbar.count >= self._total


class MaxTrackable(BaseTrackable):

    @property
    def value(self):
        return self._value

    def _to_update(self, value: float) -> bool:
        return value > self._value

    def _pass_threshold(self, value: float, threshold: float) -> bool:
        return self._value * (1.0 + threshold) <= value

    def update(self, value: float, threshold: Optional[float] = None) -> bool:
        to_update = self._to_update(value)
        pass_threshold = True
        if threshold is not None:
            pass_threshold = self._pass_threshold(value, threshold)
        if to_update:
            self._value = value
        return to_update and pass_threshold

    def reset(self):
        self._value = -float('inf')


class MinTrackable(MaxTrackable):

    def _to_update(self, value: float) -> bool:
        return value < self._value

    def _pass_threshold(self, value: float, threshold: float) -> bool:
        return self._value * (1.0 - threshold) >= value

    def reset(self):
        self._value = float('inf')


def reset_all():
    TrackableRegistry.reset_all()


class TrackableRegistry:

    _instances: Dict[str, BaseTrackable] = dict()

    def __getitem__(self, name: str) -> BaseTrackable:
        return self._instances[name]

    def __len__(self):
        return len(self._instances)

    def items(self) -> Iterator[Tuple[str, BaseTrackable]]:
        yield from self._instances.items()

    def register_trackable(self, name: str, *, total: int = None, endless: bool = False, parent: BaseTrackable = None, agg_func: str = 'count') -> BaseTrackable:
        """
        If `parent` is set, then this trackable will be reset whenever the parent is updated.
        """
        if name in self._instances:
            raise ValueError(f'A trackable named "{name}" already exists.')

        if agg_func == 'count':
            trackable = CountTrackable(name, total, endless=endless, parent=parent, registry=self)
        elif agg_func == 'max':
            trackable = MaxTrackable(name, parent=parent, registry=self)
        elif agg_func == 'min':
            trackable = MinTrackable(name, parent=parent, registry=self)
        else:
            raise ValueError(f'Unrecognized aggregate function {agg_func}.')
        self._instances[name] = trackable

        return trackable

    @classmethod
    def reset_all(cls):
        cls._instances.clear()


class TrackableUpdater:

    def __init__(self, trackable: BaseTrackable):
        self._trackable = trackable

    def update(self, *, value: Optional[Any] = None, threshold: Optional[float] = None):
        try:
            return self._trackable.update()
        except TypeError:
            return self._trackable.update(value, threshold=threshold)
