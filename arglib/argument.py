import inspect
from pathlib import Path
from typing import Any, List

ALLOWED_TYPES = [str, float, int, bool, 'path']


class DtypeNotAllowed(Exception):
    pass


class NameFormatError(Exception):
    pass


class MismatchedNArgs(Exception):
    pass


class NArgsNotAllowed(Exception):
    pass


class ChoiceNotAllowed(Exception):
    pass


class Argument:

    def __init__(
            self,
            name: str,
            *aliases,
            scope: str = None,
            dtype: Any = str,
            default: Any = None,
            nargs: Any = 1,
            msg: str = '',
            choices: List[Any] = None):
        if dtype not in ALLOWED_TYPES:
            raise DtypeNotAllowed(f'The value for "dtype" must be from {ALLOWED_TYPES}, but is actually {dtype}.')
        if not isinstance(nargs, int) and nargs != '+':
            raise NArgsNotAllowed(f'nargs can only be an int or "+", but got {nargs}.')
        if dtype == bool and nargs != 1:
            raise NArgsNotAllowed(f'For bool arguments, you can only have nargs == 1, but got {nargs}.')
        if choices is not None and not isinstance(choices, (list, tuple, set)):
            raise TypeError(f'Expect choices to be a list, tuple or set, but got {type(choices)}.')

        # Clean up name.
        name = name.strip('-').strip('_')
        if name.startswith('no_'):
            raise NameFormatError(f'Cannot have names starting with "no_", but got "f{name}".')

        # Change 'path' to Path.
        dtype = Path if dtype == 'path' else dtype

        self.name = name
        self.dtype = dtype
        self.scope = scope
        self.nargs = nargs
        self.msg = msg
        if aliases:
            self.aliases = aliases
        self.choices = None if choices is None else set(choices)
        self.value = default

    def __repr__(self):
        return f'Argument({self.name})'

    def __str__(self):
        return f'{self.name}: {self.value}'

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if new_value is None:
            self._value = new_value
        else:
            is_iterable = isinstance(new_value, (list, tuple))
            new_value_nargs = len(new_value) if is_iterable else 1
            if new_value_nargs == 1 and is_iterable:
                new_value = new_value[0]
            if self.nargs != '+' and new_value_nargs != self.nargs:
                raise MismatchedNArgs(f'self.nargs == {self.nargs}, but the new value is {new_value}.')

            if new_value_nargs == 1:
                self._value = self.dtype(new_value)
                if self.nargs == '+':
                    self._value = (self._value, )
            else:
                self._value = tuple((self.dtype(v) for v in new_value))

            # Check if the new value is in the set of choices.
            if self.choices is not None and self._value is not None:
                to_check = self._value if isinstance(self._value, tuple) else (self._value, )
                if any((v not in self.choices for v in to_check)):
                    raise ChoiceNotAllowed(f'Some value in {to_check} is not allowed. Only {self.choices} are allowed.')
