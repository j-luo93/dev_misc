from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, overload

import enlighten


class PBarOutOfBound(Exception):
    pass


class BaseTrackable(ABC):
    # IDEA(j_luo) Trackable and Tracker classes can add_trackable and put it in a registry. Maybe we can abstract some class out of it.

    def __init__(self, name: str, *, parent: Optional[BaseTrackable] = None, registry: Optional[TrackableRegistry] = None):
        self._name = name

        self.children: List[BaseTrackable] = list()
        if parent is not None:
            # NOTE(j_luo) Every child here would be reset after the parent is updated.
            parent.children.append(self)

        self.registry = registry if registry is not None else TrackableRegistry()

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
        trackable = self.registry.register_trackable(name, total=total, parent=self)
        return trackable


class CountTrackable(BaseTrackable):

    _manager = enlighten.get_manager()

    def __init__(self, name: str, total: int, **kwargs):
        self._total = total
        self._pbar = self._manager.counter(desc=name, total=total)
        super().__init__(name, **kwargs)

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
    TrackableRegistry.reset_all()


class TrackableRegistry:

    _instances: Dict[str, BaseTrackable] = dict()

    def __getitem__(self, name: str) -> BaseTrackable:
        return self._instances[name]

    def __len__(self):
        return len(self._instances)

    def register_trackable(self, name: str, *, total: int = None, parent: BaseTrackable = None, agg_func: str = 'count') -> BaseTrackable:
        """
        If `parent` is set, then this trackable will be reset whenever the parent is updated.
        """
        if name in self._instances:
            raise ValueError(f'A trackable named "{name}" already exists.')

        if agg_func == 'count':
            trackable = CountTrackable(name, total, parent=parent, registry=self)
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
        CountTrackable.reset_all()


class TrackableUpdater:

    def __init__(self, trackable: BaseTrackable):
        self._trackable = trackable

    def update(self, *, value: Any = None):
        try:
            return self._trackable.update()
        except TypeError:
            return self._trackable.update(value)
