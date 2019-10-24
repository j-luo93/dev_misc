from dataclasses import dataclass
from typing import Sequence


class DuplicateInRegistry(Exception):
    pass


class NotRegistered(Exception):
    pass


class Registry:

    def __init__(self, name):
        self._name = name
        self._instances = dict()

    @property
    def name(self):
        return self._name

    def __getitem__(self, name):
        if name not in self._instances:
            raise NotRegistered(f'Class named "{name}" not registered.')
        return self._instances[name]

    def __call__(self, cls=None, aliases: Sequence[str] = None):

        def wrapper(cls):
            names = [cls.__name__]
            cls = dataclass(cls)
            if aliases:
                names.extend(aliases)
            for name in names:
                if name in self._instances:
                    raise DuplicateInRegistry(f'Name {"name"} is already in the registry "{self._name}".')
                self._instances[name] = cls
            return cls

        if cls is None:
            return wrapper
        else:
            return wrapper(cls)
