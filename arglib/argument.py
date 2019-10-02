import inspect

ALLOWED_TYPES = [str, float, int, bool]


class DTypeNotAllowed(Exception):
    pass


class NameFormatError(Exception):
    pass


class MismatchedNArgs(Exception):
    pass


class NArgsNotAllowed(Exception):
    pass


class Argument:

    def __init__(self, name, *aliases, scope=None, dtype=str, default=None, nargs=1, msg=''):
        if dtype not in ALLOWED_TYPES:
            raise DTypeNotAllowed(f'The value for "dtype" must be from {ALLOWED_TYPES}, but is actually {dtype}.')
        if not isinstance(nargs, int) and nargs != '+':
            raise NArgsNotAllowed(f'nargs can only be an int or "+", but got {nargs}.')
        if dtype == bool and nargs != 1:
            raise NArgsNotAllowed(f'For bool arguments, you can only have nargs == 1, but got {nargs}.')

        # Clean up name.
        name = name.strip('-').strip('_')
        if name.startswith('no_'):
            raise NameFormatError(f'Cannot have names starting with "no_", but got "f{name}".')

        self.name = name
        self.dtype = dtype
        self.scope = scope
        self.nargs = nargs
        self.msg = msg
        if aliases:
            self.aliases = aliases
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
