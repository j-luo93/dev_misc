"""A FormatFile instance follows a strict convention of naming files."""

from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import List, Tuple

SUPPORTED_FILE_EXTS = {'txt', 'gz', 'pth'}
SUPPORTED_ACTIONS = {'bpe', 'tok'}


@dataclass
class LangInfo:
    lang: str = None
    pair: str = None

    def __post_init__(self):
        if not self.lang and not self.pair:
            raise RuntimeError(f'You have to specify at least one of them.')

    def __str__(self):
        nonempty = filter(lambda x: x, [self.pair, self.lang])
        return '.'.join(nonempty)


class UnsupportedError(Exception):
    pass


PROPAGATED = dict()


def propagate(func=None, *, is_iterable=False):

    @wraps(func)
    def wrapper(func):
        name = func.__name__

        @wraps(func)
        def wrapped_func(self, *args, **kwargs):
            fmts = func(self, *args, **kwargs)
            if is_iterable:
                return [FormatFile.from_fmt(fmt) for fmt in fmts]
            else:
                return FormatFile.from_fmt(fmts)

        PROPAGATED[name] = wrapped_func
        return func

    if func is None:
        return wrapper

    return wrapper(func)


@dataclass
class Format:
    folder: Path
    main: str
    lang_info: LangInfo
    ext: str
    ops: Tuple[str] = None
    part: int = None

    @property
    def lang(self):
        return self.lang_info.lang

    @property
    def pair(self):
        return self.lang_info.pair

    def __post_init__(self):
        if self.ext not in SUPPORTED_FILE_EXTS:
            raise UnsupportedError(f'{self.ext} unsupported.')
        if self.ops:
            for op in self.ops:
                if op not in SUPPORTED_ACTIONS:
                    raise UnsupportedError(f'{op} unsupported.')

    def __str__(self):
        out = str(self.folder / f'{self.main}.{self.lang_info}')
        if self.ops:
            out += '.' + '.'.join([op for op in self.ops])
        out += f'.{self.ext}'
        if self.part is not None:
            out += f'.{self.part}'
        return out

    @propagate
    def change_ext(self, new_ext: str):
        new_fmt = deepcopy(self)
        new_fmt.ext = new_ext
        return new_fmt

    @propagate
    def add_op(self, op: str):
        new_fmt = deepcopy(self)
        if self.ops:
            new_fmt.ops = self.ops + (op, )
        else:
            new_fmt.ops = (op, )
        return new_fmt

    @propagate
    def get_vocab(self):
        new_fmt = deepcopy(self)
        new_fmt.main = 'vocab'
        return new_fmt

    @propagate
    def change_folder(self, new_folder: str):
        new_fmt = deepcopy(self)
        new_fmt.folder = new_folder
        return new_fmt

    @propagate(is_iterable=True)
    def split(self, size: int):
        fmts = list()
        for part in range(size):
            fmt = Format(self.folder, self.main, self.lang_info, self.ext, self.ops, part)
            fmts.append(fmt)
        return fmts

    @propagate
    def remove_pair(self):
        new_fmt = deepcopy(self)
        new_fmt.lang_info.pair = None
        return new_fmt

    @propagate
    def remove_part(self):
        new_fmt = deepcopy(self)
        new_fmt.part = None
        return new_fmt

    @classmethod
    def extract_joint_vocab(cls, src: List['Format']):
        if len(src) != 2:
            raise RuntimeError(f'Expecting two source files but got {len(src)}.')
        f1, f2 = src
        if f1.lang == f2.lang:
            raise RuntimeError(f'Expecting two different languages, but got {f1.lang} and {f2.lang}.')
        new_fmt = deepcopy(f1)
        new_fmt.main = 'vocab'
        new_fmt.lang_info.lang = None
        if not f1.pair or not f2.pair or f1.pair != f2.pair:
            # This is for non-parallel vocab.
            new_fmt.lang_info.pair = '+'.join(sorted([f1.lang, f2.lang]))
        return new_fmt


class FormatFile:

    def __init__(self, folder: Path, main: str, lang: str, ext: str, pair: str = None, ops: Tuple[str] = None, part: int = None):
        lang_info = LangInfo(lang, pair)
        self.fmt = Format(folder, main, lang_info, ext, ops=ops, part=part)

    @property
    def path(self):
        return Path(str(self.fmt))

    @classmethod
    def from_fmt(cls, fmt: Format):
        return FormatFile(fmt.folder, fmt.main, fmt.lang_info.lang, fmt.ext, pair=fmt.lang_info.pair, ops=fmt.ops, part=fmt.part)

    @classmethod
    def extract_joint_vocab(cls, src: List['FormatFile']):
        fmts = [s.fmt for s in src]
        fmt = Format.extract_joint_vocab(fmts)
        return FormatFile.from_fmt(fmt)

    def __repr__(self):
        return str(self.fmt)

    def exists(self):
        return self.path.exists()

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError as e:
            if attr in PROPAGATED:
                method = PROPAGATED[attr]
                method = MethodType(method, self.fmt)
                return method
            raise e

    def open(self, *args, **kwargs):
        return self.path.open(*args, **kwargs)
