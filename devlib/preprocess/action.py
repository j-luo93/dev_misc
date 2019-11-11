"""An Action class that takes care of the file and its transformation."""

import logging
import os
# TODO(j_luo) Use subprocess.run
import subprocess
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union

import stanfordnlp

from .format_file import FormatFile

CONSTANTS = SimpleNamespace(
    MAIN_DIR='',
    MOSES_DIR='',
    REPLACE_UNICODE_PUNCT='',
    NORM_PUNC='',
    REM_NON_PRINT_CHAR='',
    TOKENIZER='',
    FASTBPE='',
    NUM_THREADS=8,
    EAT_DIR='',
)


class RequirementError(Exception):
    pass


class InitiateError(Exception):
    pass


class OverlappingSplit(Exception):
    pass


def _check_constant(name):
    if not getattr(CONSTANTS, name):
        raise InitiateError(f'Forgot to initiate {name}.')


def set_action_constant(name, value):
    setattr(CONSTANTS, name, value)
    if name == 'MOSES_DIR':
        CONSTANTS.REPLACE_UNICODE_PUNCT = value / 'scripts/tokenizer/replace-unicode-punctuation.perl'
        CONSTANTS.NORM_PUNC = value / 'scripts/tokenizer/normalize-punctuation.perl'
        CONSTANTS.REM_NON_PRINT_CHAR = value / 'scripts/tokenizer/remove-non-printing-char.perl'
        CONSTANTS.TOKENIZER = value / 'scripts/tokenizer/tokenizer.perl'


def _check_explicit_param(name, value):
    if value is None:
        raise RuntimeError(f'Must pass this {name} explicitly.')


def _deal_with_iterable(func=None, *, keep_single=False):
    # TODO(j_luo) add doc here. It's not very intuitive.

    def decorator(inner_func):

        @wraps(inner_func)
        def wrapper(src, *args, **kwargs):
            iter_inp = isinstance(src, (list, tuple, set))
            if iter_inp:
                ret = [inner_func(s, *args, **kwargs) for s in src]
            else:
                ret = [inner_func(src, *args, **kwargs)]
                if keep_single:
                    return ret[0]
            if isinstance(ret[0], (list, tuple, set)):
                ret = sum([list(r) for r in ret], list())
            return list(filter(lambda x: x, ret))

        return wrapper

    if func is None:
        return decorator

    return decorator(func)


@_deal_with_iterable
def _check_exists(tgt: Union[List[FormatFile], FormatFile]) -> bool:
    if not isinstance(tgt, list):
        tgt = [tgt]
    for t in tgt:
        if not t.exists():
            return False
        else:
            logging.info(f'{t} already exists.')
    return True


_Src = Union[FormatFile, str]


@_deal_with_iterable
def _get_fmt_attr(src: _Src, attr: str):
    if isinstance(src, str):
        return None
    else:
        return getattr(src.fmt, attr)


@_deal_with_iterable
def _try_mkdir(tgt: FormatFile):
    tgt.path.parent.mkdir(parents=True, exist_ok=True)


@_deal_with_iterable
def _clean_up(tgt: FormatFile):
    if tgt.exists():
        os.remove(tgt.path)


@_deal_with_iterable(keep_single=True)
def _change_folder(tgt: FormatFile, folder: Path):
    return tgt.change_folder(folder)


def _run(cmd: str):
    """Wrapper for subprocess.run"""
    subprocess.run(cmd, shell=True)


class Action(ABC):
    """
    An Action object should specify two things:
    1. Changes to the FormatFile, i.e., how does it modify the file name.
    2. The actual processing logic for the file.
    """
    REQUIRED_EXT = {}
    REQUIRED_OPS = {}

    @abstractmethod
    def change_fmt(self, src, **kwargs):
        pass

    @abstractmethod
    def act(self, src, tgt, **kwargs):
        pass

    def __call__(self, src: Union[_Src, List[_Src]], *, folder: Path = None, **kwargs):
        tgt = self.change_fmt(src, **kwargs)
        if folder is not None:
            tgt = _change_folder(tgt, folder)
        cls = type(self)
        all_fmt_exts = set(_get_fmt_attr(src, 'ext'))
        if cls.REQUIRED_EXT and len(all_fmt_exts - cls.REQUIRED_EXT) > 0:
            raise RequirementError(f'Not meeting ext requirement. {all_fmt_exts} is not in {cls.REQUIRED_EXT}.')
        all_fmt_ops = set(_get_fmt_attr(src, 'ops'))
        if cls.REQUIRED_OPS and not cls.REQUIRED_OPS <= all_fmt_ops:
            raise RequirementError(
                f'Not meeting ops requirement. {all_fmt_ops} missed something in {cls.REQUIRED_OPS}.')
        if not _check_exists(tgt):
            _try_mkdir(tgt)
            try:
                self.act(src, tgt, **kwargs)
            except subprocess.CalledProcessError as e:
                _clean_up(tgt)
                raise e
            logging.imp(f'Data saved in {tgt}.')
        return tgt


class Download(Action):

    def change_fmt(self, src: str, *, download_to: FormatFile = None, **kwargs):
        _check_explicit_param('download_to', download_to)
        return download_to

    def act(self, src: str, tgt: FormatFile, **kwargs):
        subprocess.check_call(f'wget {src} -O {tgt}', shell=True)


class Merge(Action):

    def change_fmt(self, src: List[FormatFile], *, merge_to: FormatFile = None, **kwargs):
        _check_explicit_param('merge_to', merge_to)
        return merge_to

    def act(self, src: List[FormatFile], tgt: FormatFile, **kwargs):
        all_paths = ' '.join([str(fmt_f.path) for fmt_f in src])
        _run(f'cat {all_paths} > {tgt}')


class Decompress(Action):

    REQUIRED_EXT = {'gz'}

    def change_fmt(self, src: FormatFile, **kwargs):
        return src.change_ext('txt')

    def act(self, src: FormatFile, tgt: FormatFile, **kwargs):
        subprocess.check_call(f'gunzip -c {src} > {tgt}', shell=True)


class Preprocess(Action):

    REQUIRED_EXT = {'txt'}

    def change_fmt(self, src: FormatFile, **kwargs):
        return src.add_op('tok')

    def act(self, src: FormatFile, tgt: FormatFile, **kwargs):
        _check_constant('MOSES_DIR')
        subprocess.check_call(
            f"cat {src} | {CONSTANTS.REPLACE_UNICODE_PUNCT} | {CONSTANTS.NORM_PUNC} -l {src.fmt.lang} | {CONSTANTS.REM_NON_PRINT_CHAR} | {CONSTANTS.TOKENIZER} -l {src.fmt.lang} -no-escape -threads {CONSTANTS.NUM_THREADS} > {tgt}", shell=True)


class ApplyBpe(Action):

    REQUIRED_EXT = {'txt'}
    REQUIRED_OPS = {'tok'}

    def change_fmt(self, src: FormatFile, **kwargs):
        return src.add_op('bpe')

    def act(self, src: FormatFile, tgt: FormatFile, *, codes: Path = None, **kwargs):
        if codes is None:
            raise RuntimeError(f'Must pass codes.')

        _check_constant('FASTBPE')

        # For EAT, first figure out how to deal with '<EMPTY>'.
        is_eat = 'eat' in src.fmt.ops
        if is_eat:
            empty_out = subprocess.check_output(
                f'{CONSTANTS.FASTBPE} applybpe_stream {codes} < <(echo "<EMPTY>")', shell=True, executable='/bin/bash')  # NOTE Have to use bash for this since process substitution is a bash-only feature.
            empty_out = empty_out.decode('utf8').strip()

        # Now apply BPE to everything.
        subprocess.check_call(f'{CONSTANTS.FASTBPE} applybpe {tgt} {src} {codes}', shell=True)
        if is_eat:
            subprocess.check_call(f"sed -i 's/{empty_out}/<EMPTY>/g' {tgt}", shell=True)


class Split(Action):

    REQUIRED_EXT = {'txt'}

    def change_fmt(self, src: FormatFile, *, line_ids: List[List[int]] = None, **kwargs) -> List[FormatFile]:
        _check_explicit_param('line_ids', line_ids)
        return src.split(len(line_ids))

    def act(self, src: FormatFile, tgt: List[FormatFile], *, line_ids: List[List[int]] = None, **kwargs):
        if len(tgt) != len(line_ids):
            raise RuntimeError(f'Mismatched number of splits. {len(tgt)} != {len(line_ids)}.')

        line_ids = [set(l_ids) for l_ids in line_ids]

        with src.open('r', encoding='utf8') as fin:
            fouts = [t.open('w', encoding='utf8') for t in tgt]
            for i, line in enumerate(fin):
                written = False
                for l_ids, fout in zip(line_ids, fouts):
                    if i in l_ids:
                        if written:
                            raise OverlappingSplit(f'Overlapping split for line number {i}.')
                        fout.write(line)
                        written = True
            for fout in fouts:
                fout.close()


class ExtractJointVocab(Action):

    REQUIRED_EXT = {'txt'}
    REQUIRED_OPS = {'tok'}

    def change_fmt(self, src: List[FormatFile], **kwargs):
        return FormatFile.extract_joint_vocab(src)

    def act(self, src: List[FormatFile], tgt: FormatFile, **kwargs):
        _check_constant('FASTBPE')
        src_paths = ' '.join([str(s.path) for s in src])
        subprocess.check_call(f'{CONSTANTS.FASTBPE} getvocab {src_paths} > {tgt}', shell=True)


class Binarize(Action):

    REQUIRED_EXT = {'txt'}
    REQUIRED_OPS = {'tok'}

    def change_fmt(self, src: FormatFile, **kwargs):
        return src.change_ext('pth')

    def act(self, src: FormatFile, tgt: FormatFile, *, vocab: FormatFile = None, **kwargs):
        if vocab is None:
            raise ValueError(f'Must pass vocab.')

        _check_constant('MAIN_DIR')

        subprocess.check_call(f'{CONSTANTS.MAIN_DIR}/preprocess.py {vocab} {src} {tgt}', shell=True)


class Link(Action):

    def change_fmt(self, src: FormatFile, *, link: FormatFile = None, **kwargs):
        _check_explicit_param('link', link)
        return link

    def act(self, src: FormatFile, tgt: FormatFile, **kwargs):
        tgt.path.symlink_to(src.path)


class Parse(Action):

    REQUIRED_EXT = {'txt'}
    REQUIRED_OPS = {'tok'}

    _cached_parsers = dict()

    def __init__(self, lang: str):
        self.lang = lang

    def change_fmt(self, src: FormatFile, **kwargs):
        return src.change_ext('conll')

    def act(self, src: FormatFile, tgt: FormatFile, **kwargs):
        logging.info(f'Parsing {src}...')
        if self.lang not in self._cached_parsers:
            parser = stanfordnlp.Pipeline(verbose=True, lang=self.lang, tokenize_pretokenized=True,
                                          use_gpu=True)  # This sets up a default neural pipeline.
            self._cached_parsers[self.lang] = parser
        inputs = list()
        with src.open('r', encoding='utf8') as fin:
            for line in fin:
                inputs.append(line.strip().split())
        parser = self._cached_parsers[self.lang]
        doc = parser(inputs)
        doc.write_conll_to_file(tgt.path)


class Collapse(Action):

    REQUIRED_EXT = {'conll'}
    REQUIRED_OPS = {'tok'}

    def change_fmt(self, src: FormatFile, **kwargs):
        return src.change_ext('records')

    def act(self, src: FormatFile, tgt: FormatFile, **kwargs):
        subprocess.check_call(
            f"sed ':a;N;$!ba;s/\\n\\n/@@@/g' {src} | sed ':a;N;$!ba;s/\\n/\t/g' | sed 's/@@@/\\n/g' > {tgt}", shell=True)


class ConvertEat(Action):

    REQUIRED_EXT = {'records'}
    REQUIRED_OPS = {'tok'}

    def __init__(self, *, graph: bool = False):
        self.graph = graph

    def change_fmt(self, src: FormatFile, **kwargs):
        op = 'cvtg' if self.graph else 'cvt'
        plain_tgt = src.remove_pair().add_op(op).change_ext('txt')
        eat_tgt = plain_tgt.add_op('eat')
        line_no_tgt = plain_tgt.change_ext('line_no')
        return eat_tgt, plain_tgt, line_no_tgt

    def act(self, src: FormatFile, tgt: Tuple[FormatFile], **kwargs):
        eat, plain, line_no = tgt
        logging.info(f'Converting {src} to EAT format.')
        cmd = f'python {CONSTANTS.EAT_DIR / "to_eat.py"} {src} {eat} {plain} {line_no}'
        if self.graph:
            cmd += ' --graph'
        _run(cmd)


class ConvertNeo(Action):

    REQUIRED_EXT = {'records'}
    REQUIRED_OPS = {'tok'}

    def __init__(self, *, linear: bool = False):
        self.linear = linear

    def change_fmt(self, src: FormatFile, **kwargs):
        op = 'cvtxl' if self.linear else 'cvtx'
        plain_tgt = src.remove_pair().add_op(op).change_ext('txt')
        neo_tgt = plain_tgt.add_op('neo')
        line_no_tgt = plain_tgt.change_ext('line_no')
        return neo_tgt, plain_tgt, line_no_tgt

    def act(self, src: FormatFile, tgt: Tuple[FormatFile], **kwargs):
        neo, plain, line_no = tgt
        logging.info(f'Converting {src} to NEO format.')
        cmd = f'python {CONSTANTS.EAT_DIR / "to_neo.py"} {src} {neo} {plain} {line_no}'
        if self.linear:
            cmd += ' --linear'
        _run(cmd)


class Align(Action):

    def change_fmt(self, src: List[FormatFile], **kwargs):
        return FormatFile.align(src)

    def act(self, src: List[FormatFile], tgt: List[FormatFile], *, line_nos: List[FormatFile] = None, **kwargs):
        _check_explicit_param('line_nos', line_nos)
        if not (len(line_nos) == len(src) == len(tgt) == 2):
            raise RuntimeError('Expecting to have 2 files')

        # Get parallel ids.
        ids = None
        all_line_ids = list()
        for l in line_nos:
            with l.open('r', encoding='utf8') as fl:
                l_ids = list(map(int, fl.read().split()))
                all_line_ids.append(l_ids)
            ids = set(l_ids) if ids is None else ids & set(l_ids)

        # Write to tgt.
        for s, t, l_ids in zip(src, tgt, all_line_ids):
            with s.open('r', encoding='utf8') as fs, t.open('w', encoding='utf8') as ft:
                for line, i in zip(fs, l_ids):
                    if i in ids:
                        ft.write(line)
