from dataclasses import dataclass


class DuplicateInRegistry(Exception):
    pass


class NotRegistered(Exception):
    pass


class Registry:

    def __init__(self, name):
        self._name = name
        self._instances = dict()

    def __getitem__(self, name):
        if name not in self._instances:
            raise NotRegistered(f'Class named "{name}" not registered.')
        return self._instances[name]

    def __call__(self, cls):
        cls_name = cls.__name__
        if cls_name in self._instances:
            raise DuplicateInRegistry(f'Class name {"cls_name"} is already in the registry "{self._name}".')
        cls = dataclass(cls)
        self._instances[cls_name] = cls
        return cls
