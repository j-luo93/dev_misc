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

from .trackable import (AnnealTrackable, BaseTrackable, CanAddTrackable,
                        CountTrackable, MaxTrackable, MinTrackable,
                        TrackableRegistry, TrackableUpdater)

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


class Tracker(CanAddTrackable):

    def __init__(self):
        super().__init__()
        self.settings: List[BaseSetting] = list()
        self.setting_weights: List[BaseSetting] = list()

    def is_finished(self, *names: str) -> bool:
        return all(self.registry[name].is_finished for name in names)

    def clear_trackables(self):
        """Clear all trackables. This should be called when training is done."""
        self.registry.clear_trackables()

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
            return self.registry[attr].value
        except KeyError:
            raise AttributeError(f'No trackable named {attr}.')

    def __getitem__(self, name: str) -> BaseTrackable:
        return self.registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self.registry

    def update(self, name: str, **kwargs) -> bool:
        """
        Update a trackable, and return whether it is updated. If `threshold` is provided, even if the trackable is updated,
        return False when it has not reached the threshold.
        """
        trackable = self.registry[name]
        updater = TrackableUpdater(trackable)
        return updater.update(**kwargs)

    def reset(self, *names: str):
        """Reset trackable(s) specified by name(s)."""
        for name in names:
            trackable = self.registry[name]
            trackable.reset()

    def reset_all(self):
        """Reset all trackables."""
        for name, trackable in self.registry.items():
            trackable.reset()
