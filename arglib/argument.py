import inspect

ALLOWED_TYPES = [str, float, int, bool]


class DTypeNotAllowed(Exception):
    pass


class Argument:

    def __init__(self, name, *aliases, scope=None, dtype=str, default=None):
        if dtype not in ALLOWED_TYPES:
            raise DTypeNotAllowed(f'The value for "dtype" must be from {ALLOWED_TYPES}, but is actually {dtype}.')

        self.name = name
        self.dtype = dtype
        self.scope = scope
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
            self._value = self.dtype(new_value)
