from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import enlighten


class PBarOutOfBound(Exception):
    pass


class BaseTrackable(ABC):

    def __init__(self, name: str, *, parent: BaseTrackable = None):
        self._name = name

        self.children: List[BaseTrackable] = list()
        if parent is not None:
            # NOTE(j_luo) Every child here would be reset after the parent is updated.
            parent.children.append(self)
        self.reset()

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
        trackable = TrackableFactory(name, total=total, parent=self)
        return trackable


class CountTrackable(BaseTrackable):

    _manager = enlighten.get_manager()

    def __init__(self, name: str, total: int, *, parent: BaseTrackable = None):
        self._total = total
        self._pbar = self._manager.counter(desc=name, total=total)
        super().__init__(name, parent=parent)

    @classmethod
    def reset_all(cls):
        cls._manager = enlighten.get_manager()

    @property
    def total(self):
        return self._total

    def update(self) -> bool:
        self._pbar.update()
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


class MaxTrackable(BaseTrackable):

    @property
    def value(self):
        return self._value

    def _to_update(self, value: float) -> bool:
        return value > self._value

    def update(self, value: float) -> bool:
        to_update = self._to_update(value)
        if to_update:
            self._value = value
        return to_update

    def reset(self):
        self._value = -float('inf')


class MinTrackable(MaxTrackable):

    def _to_update(self, value: float) -> bool:
        return value < self._value

    def reset(self):
        self._value = float('inf')


def reset_all():
    TrackableFactory.reset_all()


class TrackableFactory:

    _instances: Dict[str, BaseTrackable] = dict()

    def __new__(cls, name: str, *, total: int = None, parent: BaseTrackable = None, agg_func: str = 'count'):
        """
        If `parent` is set, then this trackable will be reset whenever the parent is updated.
        """
        if name in cls._instances:
            raise ValueError(f'A trackable named "{name}" already exists.')

        if agg_func == 'count':
            obj = CountTrackable(name, total, parent=parent)
        elif agg_func == 'max':
            obj = MaxTrackable(name, parent=parent)
        elif agg_func == 'min':
            obj = MinTrackable(name, parent=parent)
        else:
            raise ValueError(f'Unrecognized aggregate function {agg_func}.')

        return obj

    @classmethod
    def reset_all(cls):
        cls._instances.clear()
        CountTrackable.reset_all()


class TrackableUpdater:

    def __init__(self, trackable: BaseTrackable):
        self._trackable = trackable

    def update(self, *, value: Any = None):
        try:
            return self._trackable.update()
        except TypeError:
            return self._trackable.update(value)
