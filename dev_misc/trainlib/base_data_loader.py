from abc import ABC, abstractmethod
from typing import Dict

from torch.utils.data import DataLoader, Dataset

from .tracker.tracker import Task

from typing import Iterator


class BaseDataLoader(ABC, DataLoader):

    def __init__(self, dataset: Dataset, task: Task, **kwargs):
        self.task = task
        self._iterator: Iterator = None
        super().__init__(dataset, **kwargs)

    def get_next_batch(self):
        try:
            batch = next(self._iterator)
        except (StopIteration, TypeError):
            self._iterator = iter(self)
            batch = next(self._iterator)
        return batch


class BaseDataLoaderRegistry(ABC):

    def __init__(self):
        self._data_loaders: Dict[Task, BaseDataLoader] = dict()

    @abstractmethod
    def get_data_loader(self, task: Task, *args, **kwargs) -> BaseDataLoader:
        """Get a data loader. This is used by `register_data_loader`."""

    def register_data_loader(self, task: Task, *args, **kwargs) -> BaseDataLoader:
        if task in self._data_loaders:
            raise KeyError(f'A task named "{task}" already registered.')
        data_loader = self.get_data_loader(task, *args, **kwargs)
        self._data_loaders[task] = data_loader
        return data_loader

    def __len__(self):
        return len(self._data_loaders)

    def __getitem__(self, task: Task) -> BaseDataLoader:
        return self._data_loaders[task]
