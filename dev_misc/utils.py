from __future__ import annotations

import inspect
import warnings
from functools import reduce, wraps
from operator import iadd
from typing import ClassVar, Dict, Iterable, Iterator, Mapping, Optional

import enlighten


def check_explicit_arg(*values):
    if any(value is None for value in values):
        raise ValueError('Must explicitly pass a non-None value.')


def cached_property(func):
    """A decorator for lazy properties."""
    cached_name = f'_cached_{func.__name__}'

    @property
    @wraps(func)
    def wrapped(self):
        if not hasattr(self, cached_name):
            ret = func(self)
            setattr(self, cached_name, ret)
        return getattr(self, cached_name)

    return wrapped


def _issue_warning(func_or_cls, message: str, warning_cls):
    warnings.simplefilter("once")

    if inspect.isclass(func_or_cls):
        cls = func_or_cls
        warnings.warn(message, warning_cls)
        return cls
    else:
        func = func_or_cls

        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(message, warning_cls)
            return func(*args, **kwargs)

        return wrapped


def deprecated(func_or_cls):
    return _issue_warning(func_or_cls, f'Class/function {func_or_cls.__name__} deprecated.', DeprecationWarning)


def buggy(func_or_cls):
    return _issue_warning(func_or_cls, f'Class/function {func_or_cls.__name__} is buggy.', RuntimeWarning)


# NOTE(j_luo) This is a enlighten manager that is shared across all pbars and trackables.
manager = enlighten.get_manager()


def pbar(iterable: Iterable, desc: Optional[str] = None) -> Iterator:
    # FIXME(j_luo) Will leave the pbar hanging if there is a break in for-loop.
    try:
        total = len(iterable)
    except TypeError:
        total = None

    cnt = manager.counter(desc=desc, total=total, leave=False)
    iterator = iter(iterable)
    for item in iterator:
        yield item
        cnt.update()
    cnt.close()


class WithholdKeys:

    def __init__(self, dict_like: Mapping, *keys: str):
        self._to_track = dict_like
        self._keys = keys
        self._withheld = dict()

    def __enter__(self):
        for key in self._keys:
            self._withheld[key] = self._to_track[key]
            del self._to_track[key]

    def __exit__(self, exc_type, exc_value, traceback):
        self._to_track.update(self._withheld)


class _GlobalProperty:
    """The actual class that manages a global property."""

    _instances: ClassVar[Dict[str, _GlobalProperty]] = dict()  # pylint: disable=undefined-variable

    def __new__(cls, name: str):
        if name in cls._instances:
            obj = cls._instances[name]
        else:
            obj = super().__new__(cls)
            cls._instances[name] = obj
        return obj

    def __init__(self, name: str):
        self.name = name
        self.value = None


class global_property:
    """A descriptor class that manages how to access or set a _GlobalProperty instance."""
    # IDEA(j_luo) This can actually be used by arglib.

    def __init__(self, fget, fset=None):
        self._fget = fget
        self._g_prop = _GlobalProperty(fget.__name__)
        self._fset = fset

    def __get__(self, instance, owner=None):
        if self._g_prop.value is None:
            # NOTE(j_luo) Use ValueError so that this exception is part of the traceback when __getattr__ is supplied.
            raise ValueError(f'Global property has not been set.')
        return self._g_prop.value

    def __set__(self, instance, value):
        if self._fset is None:
            raise AttributeError(f'No setter function has been supplied.')
        self._g_prop.value = value
        self._fset(instance, value)

    def setter(self, fset):
        return type(self)(self._fget, fset=fset)


class SingletonMetaclass(type):
    """Copied from https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(metaclass=SingletonMetaclass):
    """A Singleton class that can be directly subclassed."""


def concat_lists(list_of_lists: List[list]) -> List:
    return reduce(iadd, list_of_lists, list())
