import inspect
import re
import sys
from collections import defaultdict
from functools import wraps

from pytrie import SortedStringTrie

from .argument import Argument


class ReservedNameError(Exception):
    pass


class FrozenViewError(Exception):
    pass


class DuplicateArgument(Exception):
    pass


class MultipleMatches(Exception):
    pass


class MatchNotFound(Exception):
    pass


def add_argument(name, *aliases, dtype=str, default=None, nargs=1):
    # Walk back to the frame where __qualname__ is defined.
    frame = inspect.currentframe()
    while frame is not None and '__qualname__' not in frame.f_locals:
        frame = frame.f_back
    # scope is basically the group that this argument belongs to.
    if frame is None:
        scope = 'default'
    else:
        scope = frame.f_locals['__qualname__'].split('.')[-1]
    repo = _Repository()
    repo.add_argument(name, *aliases, scope=scope, dtype=dtype, default=default, nargs=nargs)


def reset_repo():
    _Repository.reset()


def parse_args():
    repo = _Repository()
    repo.parse_args()


class _Repository:
    """Copied from https://stackoverflow.com/questions/6255050/python-thinking-of-a-module-and-its-variables-as-a-singleton-clean-approach."""

    _shared_state = {}
    _arg_trie = SortedStringTrie()

    @classmethod
    def reset(cls):
        cls._shared_state.clear()
        cls._arg_trie = SortedStringTrie()

    def __init__(self):
        self.__dict__ = self._shared_state

    def add_argument(self, name, *aliases, scope=None, dtype=str, default=None, nargs=1):
        arg = Argument(name, *aliases, scope=scope, dtype=dtype, default=default, nargs=nargs)
        if arg.name in self.__dict__:
            raise DuplicateArgument(f'An argument named "{arg.name}" has been declared.')
        if arg.name in SUPPORTED_VIEW_ATTRS:
            raise ReservedNameError(f'Name "{arg.name}" is reserved for something else.')
        self.__dict__[arg.name] = arg
        self._arg_trie[arg.name] = arg  # NOTE This is class attribute, therefore not part of __dict__.
        if dtype == bool:
            self._arg_trie[f'no_{arg.name}'] = arg

    def get_view(self):
        return _RepositoryView(self._shared_state)

    def parse_args(self):
        pattern = re.compile(r'-+[\s\w]+', re.DOTALL)
        arg_groups = re.findall(pattern, ' '.join(sys.argv))
        for group in arg_groups:
            name, *values = group.strip().split()
            name = name.strip('-')
            args = self._arg_trie.values(prefix=name)
            if len(args) > 1:
                found_names = [f'"{arg.name}"' for arg in args]
                raise MultipleMatches(f'Found more than one match for name "{name}": {", ".join(found_names)}.')
            elif len(args) == 0:
                raise MatchNotFound(f'Found no argument named "{name}".')

            arg = args[0]
            if arg.dtype == bool:
                new_value = [not name.startswith('no_')] + values
            else:
                new_value = values
            arg.value = new_value


SUPPORTED_VIEW_ATTRS = ['keys', 'values', 'items']


class _RepositoryView:

    def __init__(self, attr_dict):
        self._attr_dict = attr_dict

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            proxy = self._attr_dict
            if attr in SUPPORTED_VIEW_ATTRS:
                return getattr(proxy, attr)
            return proxy[attr].value

    def __setattr__(self, attr, value):
        if attr == '_attr_dict':
            super().__setattr__(attr, value)
        else:
            raise FrozenViewError('You cannot set values directly from here.')

    @property
    def groups(self):
        grouped = defaultdict(list)
        for arg in self._attr_dict.values():
            grouped[arg.scope].append(arg)
        for scope in grouped:
            grouped[scope].sort(key=lambda arg: arg.name)
        return {k: v for k, v in grouped.items()}


g = _Repository().get_view()


def init_g_attr(cls):

    old_init = cls.__init__

    # NOTE Properties are accessed through the class, not through the instance.
    for attr in g.keys():
        prop = property(lambda self, name=f'_{attr}': getattr(self, name).value)
        setattr(cls, attr, prop)

    @wraps(old_init)
    def new_init(self, *args, **kwargs):
        func_sig = inspect.signature(old_init)
        # First, partially bind the args and kwargs.
        bound = func_sig.bind_partial(self, *args, **kwargs)
        # Second, if possi ble, supply the missing arguments through g.
        bound_args = bound.arguments
        for k, v in g.items():
            if k not in bound_args:
                bound_args[k] = v
        # Third, apply the defaults.
        bound.apply_defaults()
        # Fourth, update the values for the private attributes (used for properties).
        for attr in g.keys():
            setattr(self, f'_{attr}', bound_args[attr])
        # Last, call the old init.
        old_init(**bound_args)

    cls.__init__ = new_init
    return cls
