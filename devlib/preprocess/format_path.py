"""
A FormatPath aims provide some structure to the paths, instead of files. The focus is on the folder structure,
instead of on the operations on individual files. So you might have some folder structure like:

folder:
    subfolder1:
        file1
        file2
    subfolder2:
        file3
        file4

Then you can specify the entire structure for each file as folder/subfolder*/file*.
"""
from pathlib import Path
from typing import List, Sequence

from .name import Name


class FormatPath:

    def __init__(self):
        self._names: List[Name] = list()

    def add(self, words: Sequence[str], fmt: str = 'snake'):
        name = Name(fmt, words)
        self._names.append(name)

    def format(self, **kwargs) -> Path:
        ret = Path(self._names[0].format(**kwargs).value)
        for name in self._names[1:]:
            ret /= name.format(**kwargs).value
        return ret
