from __future__ import annotations

from typing import Sequence, Union

import inflection


class Name:
    """A Name instance that takes care of all different forms of str."""

    def __init__(self, fmt: str, words: Union[Sequence[str], str]):
        if fmt not in ['snake', 'camel', 'hyphen']:
            raise ValueError(f'fmt can only be "snake", "hyphen" or "camel".')
        if isinstance(words, str):
            words = words.split()
        self._fmt = fmt
        self._words = words

    def format(self, **kwargs) -> Name:
        formatted = filter(lambda s: s, [word.format(**kwargs) for word in self._words])
        return Name(self._fmt, formatted)

    @property
    def value(self):
        if self._fmt == 'hyphen':
            ret = '-'.join(self._words)
        else:
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

    @property
    def hyphen(self) -> Name:
        return Name('hyphen', self._words)

    def __repr__(self):
        return f'Name("{self._name}", fmt={self._fmt})'

    def __str__(self):
        return self._name
