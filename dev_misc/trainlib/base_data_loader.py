from __future__ import annotations

from typing import Callable
from typing import overload
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from .tracker.tracker import BaseSetting


class BaseDataLoader(DataLoader):
    """Base class for DataLoader.

    Note that `collate_fn` can be specified through two means. Through a keyward argument in the constructor, or by explicitly specifying a class attribute named `collate_fn`. The former overrides the latter.
    """
    collate_fn = default_collate

    def __init__(self, dataset: Dataset, setting: BaseSetting, *args, **kwargs):
        self.setting = setting
        self._iterator: Iterator = None
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = type(self).collate_fn
        super().__init__(dataset, *args, **kwargs)

    def get_next_batch(self):
        try:
            batch = next(self._iterator)
        except (StopIteration, TypeError):
            self._iterator = iter(self)
            batch = next(self._iterator)
        return batch


class BaseDataLoaderRegistry(ABC):

    def __init__(self):
        self._data_loaders: Dict[str, BaseDataLoader] = dict()
        self._settings: Dict[str, BaseSetting] = dict()
        # self._data_loaders: Dict[Task, BaseDataLoader] = dict()

    @abstractmethod
    def get_data_loader(self, setting: BaseSetting, *args, **kwargs) -> BaseDataLoader:
        """Get a data loader. This is used by `register_data_loader`."""

    # def register_data_loader(self, task: Task, *args, **kwargs) -> BaseDataLoader:
    #     if task in self._data_loaders:
    #         raise KeyError(f'A task named "{task}" already registered.')
    #     data_loader = self.get_data_loader(task, *args, **kwargs)
    #     self._data_loaders[task] = data_loader
    #     return data_loader

    def register_data_loader(self, setting: BaseSetting, *args, **kwargs) -> BaseDataLoader:
        if setting.name in self._data_loaders:
            raise KeyError(f'A setting named "{setting}" already registered.')
        data_loader = self.get_data_loader(setting, *args, **kwargs)
        self._data_loaders[setting.name] = data_loader
        self._settings[setting.name] = setting
        return data_loader

    def __len__(self):
        return len(self._data_loaders)

    @overload
    def __getitem__(self, name: str) -> BaseDataLoader: ...

    @overload
    def __getitem__(self, setting: BaseSetting) -> BaseDataLoader: ...

    def __getitem__(self, name_or_setting):
        if isinstance(name_or_setting, str):
            return self._data_loaders[name_or_setting]
        elif isinstance(name_or_setting, BaseSetting):
            return self._data_loaders[name_or_setting.name]
        else:
            raise TypeError(f'Unsupported indexing with type "{type(name_or_setting)}".')

    def get_loaders_by_name(self, name_check: Callable[str, bool]) -> Dict[str, BaseDataLoader]:
        ret = dict()
        for name, dl in self._data_loaders.items():
            if name_check(name):
                ret[name] = dl
        return ret

    def get_setting_by_name(self, name: str) -> BaseSetting:
        return self._settings[name]
