from __future__ import annotations

import inspect
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from types import SimpleNamespace
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Union,
                    no_type_check_decorator)

from pytrie import SortedStringTrie
from typeguard import typechecked

from dev_misc.utils import deprecated

from .argument import Argument


class ReservedNameError(Exception):
    pass


class FrozenViewError(Exception):
    pass


class DuplicateArgument(Exception):
    # Set this class variable for False to disable it.
    on: ClassVar[bool] = True


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


class CheckFailed(Exception):
    """Raise this if one of the checks has failed."""


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
    """When `stacklevel` != 1, the scope will be computed based on the caller of `add_argument`."""
    scope = _get_scope(scope, stacklevel=1 + stacklevel)

    repo = _Repository()
    repo.add_argument(name, *aliases, scope=scope, dtype=dtype, default=default, nargs=nargs, msg=msg, choices=choices)


@deprecated
@dataclass
class _Condition:
    conditioner: str
    conditioner_value: Any
    conditionee: str
    conditionee_value: Any


@deprecated
def add_condition(conditioner: str, conditioner_value: Any, conditionee: str, conditionee_value: Any):
    """Add a condition between `conditioner` and `conditionee`."""
    repo = _Repository()
    condition = _Condition(conditioner, conditioner_value, conditionee, conditionee_value)
    repo.add_condition(condition)


@dataclass
class Arg:
    """A symbolic representation for an argument. Used for assertions."""
    name: str

    def __bool__(self):
        raise RuntimeError(f'You should not turn this into a boolean or evaluate this in truth conditions.')

    @typechecked
    def __eq__(self, value: _V) -> _Expression:
        return _Expression('eq', self, value)

    def __str__(self):
        return f'Arg({self.name})'


_V = Union[Arg, str, int, float]


@dataclass
class _Expression:
    op: str
    left: _VE
    right: _VE

    @typechecked
    def __or__(self, exp: _Expression) -> _Expression:
        return _Expression('or', self, exp)

    def __bool__(self):
        raise RuntimeError(f'You should not turn this into a boolean or evaluate this in truth conditions.')

    def __str__(self):
        if self.op == 'eq':
            return f'{self.left} == {self.right}'
        elif self.op == 'or':
            return f'({self.left}) | ({self.right})'
        else:
            raise ValueError(f'Unrecognized value "{self.op}" for self.op.')

    def evaluate(self, eval_func: Callable) -> bool:
        if self.op == 'eq':

            def get_value(v: _V):
                if isinstance(v, Arg):
                    return eval_func(v.name)
                return v

            left = get_value(self.left)
            right = get_value(self.right)
            return left == right
        elif self.op == 'or':
            left = self.left.evaluate(eval_func)
            right = self.right.evaluate(eval_func)
            return left or right
        else:
            raise ValueError(f'Unrecognized value "{self.op}" for self.op.')


_VE = Union[_V, _Expression]


def add_check(exp: _Expression):
    repo = _Repository()
    repo.add_check(exp)


def set_argument(name, value, *, _force=False):
    repo = _Repository()
    repo.set_argument(name, value, _force=_force)


def test_with_arguments(*, _force=False, **kwargs):
    for name, value in kwargs.items():
        # NOTE(j_luo) Use `__class__` instead of `type` since it gives us the capability of disguising the class of the value. This might come in handy in mocking.
        if isinstance(value, (list, tuple)):
            dtype = value[0].__class__
            nargs = len(value)
        else:
            dtype = value.__class__
            nargs = 1
        add_argument(name, dtype=dtype, nargs=nargs)
        set_argument(name, value, _force=_force)


def reset_repo():
    _Repository.reset()


def parse_args(known_only=False, show=False):
    """If `show is set to True, call `show_args` after parsing."""
    repo = _Repository()
    ret = repo.parse_args(known_only=known_only)
    if show:
        show_args()
    return ret


def add_registry(registry, stacklevel=1):
    repo = _Repository()
    scope = _get_scope(stacklevel=stacklevel + 1)
    repo.add_registry(registry, scope)


def get_configs():
    repo = _Repository()
    return repo.configs


def show_args(log_also=True, and_exit=False):
    _print_all_args(log_also=log_also, and_exit=and_exit)


def disable_duplicate_check():
    DuplicateArgument.on = False


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
    # IDEA(j_luo) Subclass Singleton instead.
    """Copied from https://stackoverflow.com/questions/6255050/python-thinking-of-a-module-and-its-variables-as-a-singleton-clean-approach."""

    _shared_state = dict()
    _arg_trie = SortedStringTrie()
    _registries = dict()
    _conditions: List[_Condition] = list()
    _checks: List[_Expression] = list()

    @classmethod
    def reset(cls):
        cls._shared_state.clear()
        cls._arg_trie = SortedStringTrie()
        cls._registries.clear()
        cls._conditions.clear()
        cls._checks.clear()

    def __init__(self):
        self.__dict__ = self._shared_state

    def add_argument(self, name, *aliases, scope=None, dtype=str, default=None, nargs=1, msg='', choices=None):
        if scope is None:
            raise ArgumentScopeNotSupplied('You have to explicitly set scope to a value.')

        # TODO(j_luo) Implement aliases.
        arg = Argument(name, *aliases, scope=scope, dtype=dtype, default=default, nargs=nargs, msg=msg, choices=choices)
        if arg.name in self.__dict__ and DuplicateArgument.on:
            raise DuplicateArgument(f'An argument named "{arg.name}" has been declared.')
        if arg.name in SUPPORTED_VIEW_ATTRS:
            raise ReservedNameError(f'Name "{arg.name}" is reserved for something else.')
        self.__dict__[arg.name] = arg
        self._arg_trie[arg.name] = arg  # NOTE(j_luo) This is class attribute, therefore not part of __dict__.
        if dtype == bool:
            self._arg_trie[f'no_{arg.name}'] = arg
        return arg

    @deprecated
    def add_condition(self, condition: _Condition):
        self._conditions.append(condition)

    def add_check(self, exp: _Expression):
        self._checks.append(exp)

    def set_argument(self, name, value, *, _force=False):
        if not _force:
            raise MustForceSetArgument(f'You must explicitliy set _force to True in order to set an argument.')
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

    def _get_argument_by_string(self, name, source=None) -> Argument:
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
                # TODO(j_luo) Use fields instead of vars.
                cfg = vars(cfg_cls())
                reg_source = cfg_cls.__name__
                for cfg_name, cfg_value in cfg.items():
                    cfg_arg = self._get_argument_by_string(cfg_name, source=reg_source)
                    cfg_arg.value = cfg_value
                    cfg_arg.source = reg_source
                    if cfg_name in cfg_names:
                        raise OverlappingRegistries(
                            f'Argument named "{cfg_name}" has been found in multiple registries.')
                    cfg_names.add(cfg_name)
        # Set the remaning CLI arguments.
        for arg, new_value in parsed:
            if arg.name not in self._registries:
                arg.value = new_value
                arg.source = 'CLI'
        # Check conditions (old).
        for condition in self._conditions:
            conditioner_arg = self._get_argument_by_string(condition.conditioner)
            conditionee_arg = self._get_argument_by_string(condition.conditionee)
            if conditioner_arg.value == condition.conditioner_value and conditionee_arg.value != condition.conditionee_value:
                raise ValueError(f'Condition not satisfied for {condition.conditioner} and {condition.conditionee}.')
        # Check conditions (new).
        for check in self._checks:
            truth_value = check.evaluate(lambda name: self._get_argument_by_string(name).value)
            if not truth_value:
                raise CheckFailed(f'Failed argument check for {check}.')
        return g


SUPPORTED_VIEW_ATTRS = ['keys', 'values', 'items', 'groups', '__setstate__', '__getstate__']
SUPPORTED_VIEW_MAGIC = ['__contains__', '__iter__']


def add_magic(cls):
    for mm in SUPPORTED_VIEW_MAGIC:
        setattr(cls, mm, lambda self, *args, mm=mm, **kwargs: getattr(self._attr_dict, mm)(*args, **kwargs))
    return cls


@add_magic
class _RepositoryView:

    def __init__(self, attr_dict):
        self._attr_dict = attr_dict

    def state_dict(self):
        return self._attr_dict

    def load_state_dict(self, state_dict, keep_new: bool = False):
        # FIXME(j_luo) This is very hacky. Used by pickle.
        if not hasattr(self, '_attr_dict'):
            self._attr_dict = dict()

        # Load _shared_state.
        if keep_new:
            # Keep new CLI commands.
            for k, v in state_dict.items():
                if k in self._attr_dict and self._attr_dict[k].source != 'CLI':
                    self._attr_dict[k] = v
        else:
            self._attr_dict.clear()
            self._attr_dict.update(**state_dict)
        # Load _arg_trie. Remember this is a class variable.
        for arg in self._attr_dict.values():
            _Repository._arg_trie[arg.name] = arg

    __getstate__ = state_dict
    __setstate__ = load_state_dict

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            proxy = super().__getattribute__('_attr_dict')
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
