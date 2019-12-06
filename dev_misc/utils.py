import inspect
import warnings
from functools import wraps
from typing import Iterable, Iterator, Mapping, Optional

import enlighten


def check_explicit_arg(value):
    if value is None:
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
