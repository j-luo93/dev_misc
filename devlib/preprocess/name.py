from __future__ import annotations

from typing import Sequence

import inflection


class Name:
    """A Name instance that takes care of all different forms of str."""

    def __init__(self, fmt: str, words: Sequence[str]):
        if fmt not in ['snake', 'camel']:
            raise ValueError(f'fmt can only be "snake" or "camel".')
        self._fmt = fmt
        self._words = words

    # def __hash__(self):
    #     return hash(self.canonicalize().value)

    # def __eq__(self, other):
    #     return self.canonicalize().value == other.canonicalize().value

    # def canonicalize(self) -> Name:
    #     if self._fmt == 'camel':
    #         return self.snake
    #     else:
    #         return self

    def format(self, **kwargs) -> Name:
        formatted = filter(lambda s: s, [word.format(**kwargs) for word in self._words])
        return Name(self._fmt, formatted)

    @property
    def value(self):
        ret = '_'.join(self._words)
        if self._fmt == 'camel':
            ret = inflection.camelize(ret)
        return ret

    @property
    def lowercase(self) -> Name:
        if self._fmt == 'camel':
            raise RuntimeError(f'Cannot lowercase the name in camel case.')
        new_words = [word.lower() for word in self._words]
        return Name('snake', new_words)

    @property
    def uppercase(self) -> Name:
        if self._fmt == 'camel':
            raise RuntimeError(f'Cannot uppercase the name in camel case.')
        new_words = [word.upper() for word in self._words]
        return Name('snake', new_words)

    @property
    def camel(self) -> Name:
        return Name('camel', self._words)

    @property
    def snake(self) -> Name:
        return Name('snake', self._words)

    def __repr__(self):
        return f'Name("{self._name}", fmt={self._fmt})'

    def __str__(self):
        return self._name
