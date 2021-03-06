"""
A FormatFile instance follows a strict convention of naming files.
In general, a file has one extension format (e.g., txt or gz), but could have multiple different operations done on it.
For instance, you can tokenize the file and then use BPE on it. The order of the operations should be reflected in the order
of the file name. In this case, it should `<main>.tok.bpe.txt`.
"""

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import List, Optional, Tuple

SUPPORTED_FILE_EXTS = {'txt', 'gz', 'pth', 'conll', 'records', 'line_no'}
SUPPORTED_ACTIONS = {'bpe', 'tok', 'cvt', 'eat', 'cvtx', 'neo', 'cvtxl', 'cvtg'}


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
            fmts = func(self.fmt, *args, **kwargs)
            if is_iterable:
                return [FormatFile.from_fmt(fmt) for fmt in fmts]
            else:
                return FormatFile.from_fmt(fmts)

        PROPAGATED[name] = wrapped_func
        return func

    if func is None:
        return wrapper

    return wrapper(func)


def propagate_to(cls):
    for name, func in PROPAGATED.items():
        if hasattr(cls, name):
            raise NameError(f'Class {cls.__name__} already has an attribute named {name}.')
        setattr(cls, name, func)
    return cls


def get_two_languages(src: List['Format']):
    if len(src) != 2:
        raise RuntimeError(f'Expecting two source files but got {len(src)}.')
    f1, f2 = src
    if f1.lang == f2.lang:
        raise RuntimeError(f'Expecting two different languages, but got {f1.lang} and {f2.lang}.')
    return f1, f2


@dataclass
class Format:
    folder: Path
    main: str
    lang_info: LangInfo
    ext: str
    ops: Tuple[str] = None
    part: int = None

    def clone(self):
        return deepcopy(self)

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
        new_fmt = self.clone()
        new_fmt.ext = new_ext
        return new_fmt

    @propagate
    def add_op(self, op: str):
        new_fmt = self.clone()
        if self.ops:
            new_fmt.ops = self.ops + (op, )
        else:
            new_fmt.ops = (op, )
        return new_fmt

    @propagate
    def get_vocab(self):
        new_fmt = self.clone()
        new_fmt.main = 'vocab'
        return new_fmt

    @propagate
    def change_folder(self, new_folder: str):
        new_fmt = self.clone()
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
        new_fmt = self.clone()
        new_fmt.lang_info.pair = None
        return new_fmt

    @propagate
    def remove_part(self):
        new_fmt = self.clone()
        new_fmt.part = None
        return new_fmt

    @classmethod
    def extract_joint_vocab(cls, srcs: List['Format']):
        new_fmt = srcs[0].clone()
        new_fmt.main = 'vocab'
        new_fmt.lang_info.lang = None
        new_fmt.lang_info.pair = 'joint'
        # if not f1.pair or not f2.pair or f1.pair != f2.pair:
        #     # This is for non-parallel vocab.
        #     new_fmt.lang_info.pair = '+'.join(sorted([f1.lang, f2.lang]))
        return new_fmt

    @classmethod
    def align(cls, src: List['Format']):
        f1, f2 = get_two_languages(src)
        if f1.pair or f2.pair:
            raise RuntimeError(f'Expecting both sources to be non-parallel, but got {f1.pair} and {f2.pair}.')
        new_pair = '-'.join(sorted([f1.lang, f2.lang]))
        new_fmt1 = f1.clone()
        new_fmt1.lang_info.pair = new_pair
        new_fmt2 = f2.clone()
        new_fmt2.lang_info.pair = new_pair
        return new_fmt1, new_fmt2


@propagate_to
class FormatFile:

    def __init__(self, folder: Path, main: str, lang: str, ext: str, *, pair: Optional[str] = None, ops: Optional[Tuple[str]] = None, part: Optional[int] = None, compat_main: Optional[str] = None):
        """
        `compat_main` is used by files that do not follow the name format conventions.
        """
        lang_info = LangInfo(lang, pair)
        self.fmt = Format(folder, main, lang_info, ext, ops=ops, part=part)
        self.compat_main = compat_main

    @property
    def path(self):
        if self.compat_main is None:
            return Path(str(self.fmt))
        else:
            return Path(self.fmt.folder / self.compat_main)

    @classmethod
    def from_fmt(cls, fmt: Format):
        return FormatFile(fmt.folder, fmt.main, fmt.lang_info.lang, fmt.ext, pair=fmt.lang_info.pair, ops=fmt.ops, part=fmt.part)

    @classmethod
    def extract_joint_vocab(cls, src: List['FormatFile']):
        fmts = [s.fmt for s in src]
        fmt = Format.extract_joint_vocab(fmts)
        return FormatFile.from_fmt(fmt)

    @classmethod
    def align(cls, src: List['FormatFile']):
        fmts = [s.fmt for s in src]
        fmts = Format.align(fmts)
        return [FormatFile.from_fmt(fmt) for fmt in fmts]

    def __repr__(self):
        return str(self.fmt)

    def exists(self):
        return self.path.exists()

    def remove(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass

    def open(self, *args, **kwargs):
        return self.path.open(*args, **kwargs)
