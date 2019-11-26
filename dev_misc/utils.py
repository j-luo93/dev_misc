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
