from __future__ import annotations

import inspect
import logging
import sys
import warnings
from functools import lru_cache, reduce, wraps
from operator import iadd, ior
from typing import (Callable, ClassVar, Dict, Iterable, Iterator, List,
                    Mapping, Optional)

import enlighten


def check_explicit_arg(*values):
    if any(value is None for value in values):
        raise ValueError('Must explicitly pass a non-None value.')


def cached_property(_func=None, *, in_class: bool = False, key=None):
    """A decorator for lazy properties.

    If `in_class` is True, the cache is stored in class instead of in instances.
    If `key` is provided, the cache will be a dict with `key(self)` as the actual key.
    """

    def wrap(_func):

        cache_name = f'_cached_{_func.__name__}'
        if key is not None and not in_class:
            logging.warn(f'`key` ignored since `in_class` is set to False.')

        @property
        @wraps(_func)
        def wrapped(self):
            cache_target = type(self) if in_class else self
            if key is not None and in_class:
                # Stored as part of a dict which is stored as an attribute.
                if not hasattr(cache_target, cache_name):
                    setattr(cache_target, cache_name, dict())
                cache = getattr(cache_target, cache_name)
                k = key(self)
                if k not in cache:
                    v = _func(self)
                    cache[k] = v
                return cache[k]
            else:
                # Stored as an attribute.
                if not hasattr(cache_target, cache_name):
                    v = _func(self)
                    setattr(cache_target, cache_name, v)
                return getattr(cache_target, cache_name)

        return wrapped

    if _func is None:
        return wrap

    return wrap(_func)


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


def pbar(iterable: Iterable, desc: Optional[str] = None, text_only: bool = False) -> Iterator:
    """If `text_only` is True, use sys.stdout and print as usual, otherwise use `enlighten`."""
    # FIXME(j_luo) Will leave the pbar hanging if there is a break in for-loop.
    try:
        total = len(iterable)
    except TypeError:
        total = None

    iterator = iter(iterable)

    if text_only:
        for i, item in enumerate(iterator, 1):
            yield item
            print(f'\r{i}', end='')
            sys.stdout.flush()
        print()
    else:
        cnt = manager.counter(desc=desc, total=total, leave=False)
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


def concat_lists(list_of_lists: List[list]) -> list:
    return reduce(iadd, list_of_lists, list())


def union_sets(list_of_sets: List[set]) -> set:
    return reduce(ior, list_of_sets, set())


def handle_sequence_inputs(_func):
    """A decorator for functions to enable them to handle sequence inputs. Currently only support single-argument functions."""

    sign = inspect.signature(_func)
    if not inspect.isfunction(_func):
        raise TypeError(f'Can only support functions.')
    if len(sign.parameters) != 1:
        raise TypeError(f'Can only support single-argument functions.')

    @wraps(_func)
    def wrapped(seq):
        return [_func(x) for x in seq]

    return wrapped


class _CacheSwitches:

    def __init__(self):
        self._switch_status: Dict[str, bool] = dict()
        self._switch2func: Dict[str, Callable] = dict()

    def __getitem__(self, switch: str) -> bool:
        return self._switch_status[switch]

    def add_switch(self, switch: str, func: Callable):
        if switch in self._switch_status:
            raise NameError(f'A switch named {switch} already exists.')

        self._switch_status[switch] = False
        self._switch2func[switch] = func

    def switch_on(self, switch: str):
        if self._switch_status[switch]:
            raise RuntimeError(f'The switch named {switch} has already been turned on.')
        self._switch_status[switch] = True

    def switch_off(self, switch: str):
        if not self._switch_status[switch]:
            raise RuntimeError(f'The switch named {switch} has already been turned off.')
        self._switch_status[switch] = False
        func = self._switch2func[switch]
        func.cache_clear()


_cache_switches = _CacheSwitches()


def cacheable(_func=None, *, switch: Optional[str] = None):
    if switch is None:
        switch = _func.__name__

    def wrap(_func):
        _cacheable_func = lru_cache(maxsize=None)(_func)
        _cache_switches.add_switch(switch, _cacheable_func)

        @wraps(_func)
        def wrapped(self, *args, **kwargs):
            func_to_run = _cacheable_func if _cache_switches[switch] else _func
            return func_to_run(self, *args, **kwargs)

        return wrapped

    if _func is None:
        return wrap

    return wrap(_func)


class ScopedCache:

    def __init__(self, switch: str):
        self._switch = switch

    def __enter__(self):
        _cache_switches.switch_on(self._switch)

    def __exit__(self, exc_type, exc_value, traceback):
        _cache_switches.switch_off(self._switch)
