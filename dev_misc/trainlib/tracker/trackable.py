from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import (Any, ClassVar, Dict, Iterator, List, Optional, Tuple,
                    overload)

from dev_misc.utils import is_main_process_and_thread, manager


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
        # Only add a pbar in main process. Otherwise use a simpler counter.
        if is_main_process_and_thread():
            self._pbar = manager.counter(desc=name, total=total, leave=False)
        else:
            self._count = 0
        super().__init__(name, **kwargs)

    @property
    def total(self):
        return self._total

    def update(self) -> bool:
        if self.value == self.total and self._endless:
            self.reset()

        while True:
            try:
                if is_main_process_and_thread():
                    self._pbar.update()
                else:
                    self._count += 1
                break
            except RuntimeError:
                logging.exception('Encountered some issue when updating the progress bar. Waiting to retry again')
                time.sleep(1)

        if self._total is not None and self.value > self._total:
            raise PBarOutOfBound(f'Progress bar ran out of bound.')
        for trackable in self.children:
            trackable.reset()
        return True

    def reset(self):
        if is_main_process_and_thread():
            self._pbar.start = time.time()
            self._pbar.count = 0
            self._pbar.refresh()
        else:
            self._count = 0

    @property
    def value(self):
        if is_main_process_and_thread():
            return self._pbar.count
        else:
            return self._count

    @property
    def is_finished(self) -> bool:
        return self._total and self.value >= self._total

    def close(self):
        """Remove progress bar."""
        # FIXME(j_luo) It seems that enlighten is still buggy?
        if is_main_process_and_thread():
            manager.remove(self._pbar)


class AnnealTrackable(BaseTrackable):

    def __init__(self, name: str, init_value: float, multiplier: float, bound: float, **kwargs):
        self._init_value = init_value
        self._multiplier = multiplier
        self._bound = bound
        # Only add a pbar in main process. Otherwise use a simpler counter.
        if is_main_process_and_thread():
            self._status_bar = manager.status_bar(status_format=f'Anneal: {name} ' + '{value:.5f}', value=init_value)
        super().__init__(name, **kwargs)

    @property
    def value(self):
        return self._value

    def reset(self):
        self._value = self._init_value

    def update(self):
        new_value = self._multiplier * self._value
        if self._multiplier >= 1.0:
            self._value = min(new_value, self._bound)
        else:
            self._value = max(new_value, self._bound)
        self._status_bar.update(value=self._value)

    def close(self):
        if is_main_process_and_thread():
            manager.remove(self._status_bar)


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

    _instances: ClassVar[Dict[str, BaseTrackable]] = dict()

    def __init__(self):
        self._trackables = dict()

    def __getitem__(self, name: str) -> BaseTrackable:
        return self._trackables[name]

    def __contains__(self, name: str) -> bool:
        return name in self._trackables

    def __len__(self):
        return len(self._trackables)

    def items(self) -> Iterator[Tuple[str, BaseTrackable]]:
        yield from self._trackables.items()

    def register_trackable(self, name: str, *,
                           total: int = None,
                           endless: bool = False,
                           init_value: float = None,
                           multiplier: float = None,
                           bound: float = None,
                           parent: BaseTrackable = None,
                           agg_func: str = 'count') -> BaseTrackable:
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
        elif agg_func == 'anneal':
            trackable = AnnealTrackable(name, init_value, multiplier, bound, parent=parent, registry=self)
        else:
            raise ValueError(f'Unrecognized aggregate function {agg_func}.')
        self._trackables[name] = trackable
        self._instances[name] = trackable

        return trackable

    @classmethod
    def reset_all(cls):
        cls._instances.clear()

    def clear_trackables(self):
        for name, trackable in self._trackables.items():
            try:
                trackable.close()
            except AttributeError:
                pass
        for name in list(self._trackables.keys()):
            del self._trackables[name]
            del self._instances[name]


class TrackableUpdater:

    def __init__(self, trackable: BaseTrackable):
        self._trackable = trackable

    def update(self, *, value: Optional[Any] = None, threshold: Optional[float] = None):
        try:
            return self._trackable.update()
        except TypeError:
            return self._trackable.update(value, threshold=threshold)
