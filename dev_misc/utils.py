import inspect
import warnings
from functools import wraps


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


def _warn_once(message: str, exception_cls: Warning):
    with warnings.catch_warnings():
        warnings.simplefilter("once")
        warnings.warn(message, exception_cls)


def deprecated(func_or_cls):

    if inspect.isclass(func_or_cls):
        cls = func_or_cls
        _warn_once(f'Class {cls.__name__} deprecated.', DeprecationWarning)
        return cls
    else:
        func = func_or_cls

        @wraps(func)
        def wrapped(*args, **kwargs):
            _warn_once(f'Function {func.__name__} deprecated.', DeprecationWarning)
            return func(*args, **kwargs)

        return wrapped