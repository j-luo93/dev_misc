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
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type

from dev_misc.utils import deprecated

from .trackable import (AnnealTrackable, BaseTrackable, CountTrackable,
                        MaxTrackable, MinTrackable, TrackableRegistry,
                        TrackableUpdater)

# setting_class: Type[dataclass] = update_wrapper(partial(dataclass, eq=False), dataclass)


@dataclass
class BaseSetting:
    """A setting is just an umbrella term that includes tasks, data loader setups and etc.

    A given setting determines a data loader object but not the other way around -- one data loader might be
    used in multiple settings.
    """
    name: str

    def __hash__(self):
        return id(self.name)

    def __eq__(self, other: BaseSetting):
        return self.name == other.name


class Tracker:

    def __init__(self):
        self.settings: List[BaseSetting] = list()
        self.setting_weights: List[BaseSetting] = list()

        # A centralized registry for all trackables.
        self.trackable_reg = TrackableRegistry()

    def is_finished(self, *names: str) -> bool:
        return all(self.trackable_reg[name].is_finished for name in names)

    def add_trackable(self, name: str, **kwargs) -> BaseTrackable:
        trackable = self.trackable_reg.register_trackable(name, **kwargs)
        return trackable

    def clear_trackables(self):
        """Clear all trackables. This should be called when training is done."""
        self.trackable_reg.clear_trackables()

    def add_max_trackable(self, name: str) -> MaxTrackable:
        return self.add_trackable(name, agg_func='max')

    def add_min_trackable(self, name: str) -> MinTrackable:
        return self.add_trackable(name, agg_func='min')

    def add_count_trackable(self, name: str, total: int) -> CountTrackable:
        return self.add_trackable(name, total=total, agg_func='count')

    def add_anneal_trackable(self, name: str, init_value: float, multiplier: float, bound: float) -> AnnealTrackable:
        return self.add_trackable(name, init_value=init_value,
                                  multiplier=multiplier, bound=bound, agg_func='anneal')

    def add_setting(self, setting: BaseSetting, weight: float):
        self.settings.append(setting)
        self.setting_weights.append(weight)

    def add_settings(self, settings: Sequence[BaseSetting], weights: List[float]):
        if len(settings) != len(weights):
            raise ValueError(f'Mismatched lengths from settings ({len(settings)}) and weights ({len(weights)}).')
        for setting, weight in zip(settings, weights):
            self.add_setting(setting, weight)

    def draw_setting(self) -> BaseSetting:
        setting = random.choices(self.settings, weights=self.setting_weights)[0]
        return setting

    def __getattr__(self, attr: str):
        try:
            return self.trackable_reg[attr].value
        except KeyError:
            raise AttributeError(f'No trackable named {attr}.')

    def __getitem__(self, name: str) -> BaseTrackable:
        return self.trackable_reg[name]

    def __contains__(self, name: str) -> bool:
        return name in self.trackable_reg

    def update(self, name: str, *, value: Optional[Any] = None, threshold: Optional[float] = None) -> bool:
        """
        Update a trackable, and return whether it is updated. If `threshold` is provided, even if the trackable is updated,
        return False when it has not reached the threshold.
        """
        trackable = self.trackable_reg[name]
        updater = TrackableUpdater(trackable)
        return updater.update(value=value, threshold=threshold)

    def reset(self, *names: str):
        """Reset trackable(s) specified by name(s)."""
        for name in names:
            trackable = self.trackable_reg[name]
            trackable.reset()

    def reset_all(self):
        """Reset all trackables."""
        for name, trackable in self.trackable_reg.items():
            trackable.reset()
