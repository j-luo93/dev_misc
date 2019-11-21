import operator
from functools import wraps
from typing import Any, Callable

from .parser import g


def _g_decorator_helper(name: str, value: Any = None, *, when: str = 'call', callback: Callable = None, comp: str = 'eq', else_return='original'):
    """The backend call for many g-based decorators."""
    assert when in ['pre', 'post', 'call']
    assert else_return in ['original', 'none']
    if when in ['pre', 'post']:
        assert callback is not None

    def _is_met(first_value, second_value):
        try:
            op = getattr(operator, comp)
        except AttributeError:
            raise ValueError(f'Comparison operator not found.')
        return op(first_value, second_value)

    def wrapper(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            # Set default return.
            if else_return == 'original':
                if kwargs:
                    raise NotImplementedError(f'Cannot deal with nonempty kwargs right now.')
                if len(args) == 1:
                    ret = args[0]
                else:
                    ret = args
            else:
                ret = None

            g_value = getattr(g, name)
            if value is None:
                met = bool(g_value)
            else:
                met = _is_met(g_value, value)

            if when == 'pre' and met:
                callback()
            if when != 'call' or met:
                ret = func(*args, **kwargs)
            if when == 'post' and met:
                callback()
            return ret

        return wrapped

    return wrapper


def not_supported_argument_value(name: str, value: Any = None, comp: str = 'eq'):
    """A decorator that will raise error if the argument has a certain value."""

    def callback():
        raise NotImplementedError(f'Argument "{name}" with value "{value}" not supported.')

    return _g_decorator_helper(name, value, when='pre', callback=callback, comp=comp)


def try_when(name: str, value: Any = None, comp: str = 'eq'):
    """A decorator that will only run the wrapped function if the argument has a certain value."""

    return _g_decorator_helper(name, value, comp=comp)
