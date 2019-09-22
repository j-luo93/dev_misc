import inspect
from collections import defaultdict

from .argument import Argument


class FrozenViewError(Exception):
    pass


class DuplicateArgument(Exception):
    pass


def add_argument(name, *aliases, dtype=str, default=None):
    # Walk back to the frame where __qualname__ is defined.
    frame = inspect.currentframe()
    while frame is not None and '__qualname__' not in frame.f_locals:
        frame = frame.f_back
    if frame is None:
        scope = None
    else:
        scope = frame.f_locals['__qualname__']
    repo = _Repository()
    repo.add_argument(name, *aliases, scope=scope, dtype=dtype, default=default)


def reset_repo():
    _Repository.reset()


class _Repository:
    """Copied from https://stackoverflow.com/questions/6255050/python-thinking-of-a-module-and-its-variables-as-a-singleton-clean-approach."""

    _shared_state = dict()

    @classmethod
    def reset(cls):
        cls._shared_state.clear()

    def __init__(self):
        self.__dict__ = self._shared_state

    def add_argument(self, name, *aliases, scope=None, dtype=str, default=None):
        arg = Argument(name, *aliases, scope=scope, dtype=dtype, default=default)
        if name in self.__dict__:
            raise DuplicateArgument(f'An argument named "{name}" has been declared.')
        self.__dict__[name] = arg

    def get_view(self):
        return _RepositoryView(self._shared_state)


class _RepositoryView:

    def __init__(self, attr_dict):
        self._attr_dict = attr_dict

    def __getattr__(self, attr):
        return self._attr_dict[attr].value

    def __setattr__(self, attr, value):
        if attr == '_attr_dict':
            super().__setattr__(attr, value)
        else:
            raise FrozenViewError('You cannot set values directly from here.')


g = _Repository().get_view()
