import inspect
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from types import SimpleNamespace
from typing import Any, Dict, no_type_check_decorator

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


class DuplicateRegistry(Exception):
    pass


class OverlappingRegistries(Exception):
    pass


class MustForceSetArgument(Exception):
    pass


class ArgumentScopeNotSupplied(Exception):
    pass


def _get_scope(scope=None, stacklevel=1):
    # TODO(j_luo) rename scope to sth else?
    """
    scope is basically the group that this argument belongs to.
    """
    # Get the proper frame according to stacklevel first.
    if scope is None:
        frame = inspect.currentframe()
        for _ in range(stacklevel):
            frame = frame.f_back
        # NOTE(j_luo) This means that it's within class definition.
        module_name = frame.f_globals['__name__']
        if '__qualname__' in frame.f_locals:
            qualname = frame.f_locals['__qualname__']
            scope = '.'.join([module_name, qualname])
        else:
            scope = module_name

    return scope


def add_argument(name, *aliases, dtype=str, default=None, nargs=1, msg='', choices=None, scope=None, stacklevel=1):
    """When stacklevel == 1, the scope will be computed based on the caller of `add_argument`."""
    scope = _get_scope(scope, stacklevel=1 + stacklevel)

    repo = _Repository()
    repo.add_argument(name, *aliases, scope=scope, dtype=dtype, default=default, nargs=nargs, msg=msg, choices=choices)


def set_argument(name, value, *, force=False):
    repo = _Repository()
    repo.set_argument(name, value, force=force)


def reset_repo():
    _Repository.reset()


def parse_args(known_only=False):
    repo = _Repository()
    return repo.parse_args(known_only=known_only)


def add_registry(registry, stacklevel=1):
    repo = _Repository()
    scope = _get_scope(stacklevel=stacklevel + 1)
    repo.add_registry(registry, scope)


def get_configs():
    repo = _Repository()
    return repo.configs


def show_args():
    _print_all_args()


def _print_all_args(log_also=True, and_exit=False):
    output = ''
    for group, args in g.groups.items():
        output += f'{group}:\n'
        for arg in args:
            output += f'\t{arg}\n'
    output = output.strip()
    if log_also:
        logging.info(output)
    else:
        print(output)
    if and_exit:
        sys.exit()


class _Repository:
    """Copied from https://stackoverflow.com/questions/6255050/python-thinking-of-a-module-and-its-variables-as-a-singleton-clean-approach."""

    _shared_state = dict()
    _arg_trie = SortedStringTrie()
    _registries = dict()

    @classmethod
    def reset(cls):
        cls._shared_state.clear()
        cls._arg_trie = SortedStringTrie()
        cls._registries.clear()

    def __init__(self):
        self.__dict__ = self._shared_state

    def add_argument(self, name, *aliases, scope=None, dtype=str, default=None, nargs=1, msg='', choices=None):
        if scope is None:
            raise ArgumentScopeNotSupplied('You have to explicitly set scope to a value.')

        arg = Argument(name, *aliases, scope=scope, dtype=dtype, default=default, nargs=nargs, msg=msg, choices=choices)
        if arg.name in self.__dict__:
            raise DuplicateArgument(f'An argument named "{arg.name}" has been declared.')
        if arg.name in SUPPORTED_VIEW_ATTRS:
            raise ReservedNameError(f'Name "{arg.name}" is reserved for something else.')
        self.__dict__[arg.name] = arg
        self._arg_trie[arg.name] = arg  # NOTE(j_luo) This is class attribute, therefore not part of __dict__.
        if dtype == bool:
            self._arg_trie[f'no_{arg.name}'] = arg
        return arg

    def set_argument(self, name, value, *, force=False):
        if not force:
            raise MustForceSetArgument(f'You must explicitliy set force = True in order to set an argument.')
        arg = self._get_argument_by_string(name)
        arg.value = value

    def add_registry(self, registry, scope):
        try:
            arg = self.add_argument(registry.name, scope=scope, dtype=str)
        except DuplicateArgument:
            raise DuplicateRegistry(f'A registry named "{registry.name}" already exists.')
        self._registries[arg.name] = registry

    @property
    def configs(self) -> Dict[str, str]:
        ret = dict()
        for name, registry in self._registries.items():
            arg = self._get_argument_by_string(name)
            ret[name] = arg.value
        return ret

    def get_view(self):
        return _RepositoryView(self._shared_state)

    def _get_argument_by_string(self, name, source='CLI'):
        args = self._arg_trie.values(prefix=name)
        if len(args) > 1:
            found_names = [f'"{arg.name}"' for arg in args]
            raise MultipleMatches(f'Found more than one match for name "{name}": {", ".join(found_names)}.')
        elif len(args) == 0:
            raise MatchNotFound(f'Found no argument named "{name}" from "{source}".')
        arg = args[0]
        return arg

    def parse_args(self, known_only=False):
        arg_groups = list()
        group = list()
        for seg in sys.argv[1:]:
            if seg.startswith('-'):
                if group:
                    arg_groups.append(group)
                group = seg.split('=')
            else:
                group.append(seg)
        if group:
            arg_groups.append(group)
        # Parse the CLI string first.
        parsed = list()
        for group in arg_groups:
            name, *values = group
            name = name.strip('-')
            # NOTE(j_luo) Help mode. Note that if known_only is True, then help is ignored.
            if name == 'h' or name == 'help':
                if not known_only:
                    _print_all_args(log_also=False, and_exit=True)
                continue  # NOTE(j_luo) Some other args might start with "h"!
            try:
                arg = self._get_argument_by_string(name, source='CLI')
                if arg.dtype == bool:
                    new_value = [not name.startswith('no_')] + values
                else:
                    new_value = values
                parsed.append((arg, new_value))
            except MatchNotFound as e:
                if not known_only:
                    raise e
        # Deal with config files and use their values to set the new default values.
        cfg_names = set()
        for arg, new_value in parsed:
            if arg.name in self._registries:
                arg.value = new_value
                reg = self._registries[arg.name]
                cfg_cls = reg[arg.value]
                cfg = vars(cfg_cls())
                for cfg_name, cfg_value in cfg.items():
                    cfg_arg = self._get_argument_by_string(cfg_name, source=cfg_cls.__name__)
                    cfg_arg.value = cfg_value
                    if cfg_name in cfg_names:
                        raise OverlappingRegistries(
                            f'Argument named "{cfg_name}" has been found in multiple registries.')
                    cfg_names.add(cfg_name)
        # Set the remaning CLI arguments.
        for arg, new_value in parsed:
            if arg.name not in self._registries:
                arg.value = new_value
        return g


SUPPORTED_VIEW_ATTRS = ['keys', 'values', 'items']
SUPPORTED_VIEW_MAGIC = ['__contains__', '__iter__']


def add_magic(cls):
    for mm in SUPPORTED_VIEW_MAGIC:
        setattr(cls, mm, lambda self, *args, mm=mm, **kwargs: getattr(self._attr_dict, mm)(*args, **kwargs))
    return cls


# IDEA(j_luo) Try subclassing dict directly or use a proxy?
@add_magic
class _RepositoryView:

    def __init__(self, attr_dict):
        self._attr_dict = attr_dict

    def state_dict(self):
        return self._attr_dict

    def load_state_dict(self, state_dict):
        # Load _shared_state.
        self._attr_dict.clear()
        self._attr_dict.update(**state_dict)
        # Load _arg_trie. Remember this is a class variable.
        for arg in self._attr_dict.values():
            _Repository._arg_trie[arg.name] = arg

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
        for arg in self.values():
            grouped[arg.scope].append(arg)
        for scope in grouped:
            grouped[scope].sort(key=lambda arg: arg.name)
        return {k: v for k, v in grouped.items()}

    def as_dict(self):
        return {k: arg.value for k, arg in self.items()}

    def as_namespace(self):
        return SimpleNamespace(**self.as_dict())


g = _Repository().get_view()


def not_supported_argument_value(name: str, value: Any):
    """A decorator that will raise error if the argument has a certain value."""

    def wrapper(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            if g.name == value:
                raise NotImplementedError(f'Argument "{name}" with value "{value}" not supported.')
            return func(*args, **kwargs)

        return func

    return wrapper


ALLOWED_INIT_G_ATTR_DEFAULT = ['property', 'none', 'attribute']


# IDEA(j_luo) Check out this https://docs.python.org/3/library/typing.html#typing.no_type_check
# FIXME(j_luo) This would break down if annotations is imported from __future__.
def init_g_attr(cls=None, *, default='none'):
    """The signature and the main body of this function follow `dataclass` in https://github.com/python/cpython/blob/master/Lib/dataclasses.py.
    But positional-only marker "/" is removed since it is not supported in Python 3.7 yet.

    Use @init_g_attr to @init_g_attr(default='property') to set everything (apart from self) as properties.
    There are other possible values for "default":
    1. "none": This will return to the original default of not doing anything.
    2. "attr": This will add the argument as a normal attribute.
    All non-default actions for the arguments should be annotated.

    Default is set to "none" to avoid confusion regarding attributes and properties.
    """
    if default not in ALLOWED_INIT_G_ATTR_DEFAULT:
        raise ValueError(
            f'The value for "default" must be from {ALLOWED_INIT_G_ATTR_DEFAULT}, but is actually {default}.')

    def wrap(cls):
        old_init = cls.__init__

        # Analyze every argument in the signature to figure out what to do with each of them.
        sig = inspect.signature(old_init)
        params = sig.parameters
        actions = dict()
        type2action = {
            'p': 'property',
            'n': 'none',
            'a': 'attribute'
        }
        for name, param in params.items():
            if name == 'self':
                actions[name] = 'none'
            elif param.annotation == inspect.Signature.empty:
                actions[name] = default
            else:
                try:
                    actions[name] = type2action[param.annotation]
                except KeyError:
                    raise KeyError(f'Annotation type "{param.annotation}" not supported.')

        # NOTE(j_luo) Properties are accessed through the class, not through the instance. Hence they should be set up
        # prior to __init__ calls.
        for name, action in actions.items():
            if action == 'property':
                prop = property(lambda self, name=f'_{name}': getattr(self, name))
                setattr(cls, name, prop)

        @wraps(old_init)
        def new_init(self, *args, **kwargs):
            # First, partially bind the args and kwargs.
            bound = sig.bind_partial(self, *args, **kwargs)
            # Second, if possible, supply the missing arguments through g. Note that this is in-place.
            already_bound = bound.arguments
            for attr in params:
                if attr not in already_bound and attr in g:
                    bound.arguments[attr] = getattr(g, attr)
            # Third, apply the defaults.
            bound.apply_defaults()
            # Fourth, add attributes and finish setting up properties.
            for name, action in actions.items():
                if action == 'property':
                    setattr(self, f'_{name}', bound.arguments[name])
                elif action == 'attribute':
                    setattr(self, name, bound.arguments[name])
            # Last, call the old init.
            old_init(**bound.arguments)

        cls.__init__ = new_init
        return cls

    if cls is None:
        return wrap

    return wrap(cls)
