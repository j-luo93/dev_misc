import random
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, Hashable, List, Union

from deprecated import deprecated

from .action import (Align, ApplyBpe, Binarize, Collapse, ConvertEat,
                     ConvertNeo, Decompress, Download, ExtractJointVocab, Link,
                     Merge, Parse, Preprocess, Split)
from .format_file import FormatFile

Sources = Dict[Hashable, Union[FormatFile, str]]


def _apply_action(action_cls):

    @wraps(action_cls.__call__)
    def wrapper(self, **kwargs):
        action = action_cls()
        for k in self.sources:
            self.sources[k] = action(self.sources[k], **kwargs)

    return wrapper


@dataclass
class EatAuxFiles:
    plain: FormatFile
    line_no: FormatFile


@dataclass
class NeoAuxFiles:
    neo: FormatFile
    line_no: FormatFile


class Pipeline:

    def __init__(self, sources: Sources):
        self.sources = sources
        self.vocab = None
        self.eat_aux_files = dict()
        self.neo_aux_files = dict()

    def download(self, **common_info):
        download_to = dict()
        for k in self.sources:
            info = deepcopy(common_info)
            # TODO(j_luo) This is making assumptions about the key (`k`) that doesn't necessarily hold true.
            info.update(k._asdict())
            fmt_file = FormatFile(**info)
            download_to[k] = fmt_file
        action = Download()
        for k in self.sources:
            action(self.sources[k], download_to=download_to[k])
        self.sources = download_to

    def merge(self, to_merge_keys: List[Hashable], merge_to_key: Hashable, merge_to: FormatFile):
        to_merge = [self.sources[key] for key in to_merge_keys]
        action = Merge()
        action(to_merge, merge_to=merge_to)
        for key in to_merge_keys:
            del self.sources[key]
        self.sources[merge_to_key] = merge_to

    def split(self, to_split_key: Hashable, line_ids: List[List[int]], take: int):
        action = Split()
        self.sources[to_split_key] = action(self.sources[to_split_key], line_ids=line_ids)[take]

    def split_n_parts(self, to_split_keys: List[Hashable], n_parts: int, take: List[int]):
        # TODO(j_luo) make getting line count lazy? maybe we can move a lot of this to action.py
        if not isinstance(to_split_keys, list) or not isinstance(take, list):
            raise TypeError(
                f'Expect both to_split_keys and take to be a list, but got {type(to_split_keys)} and {type(take)} respectively.')
        if len(to_split_keys) != len(take):
            raise ValueError(
                f'Expect to_split_keys and take to have the same length, but got {len(to_split_keys)} and {len(take)} respectively.')

        all_res = list()
        for key in to_split_keys:
            path = self.sources[key].path
            res = subprocess.run(f'cat {path} | wc -l', capture_output=True, shell=True)
            if res.returncode != 0:
                raise OSError('Cannot obtain line count for some reason.')
            all_res.append(int(res.stdout.decode('utf8')))
        if any([other != all_res[0] for other in all_res[1:]]):
            raise RuntimeError(f'Files with key {to_split_keys} do not have the same line counts.')

        num_lines = all_res[0]
        indices = list(range(num_lines))
        random.shuffle(indices)
        lines_per_part = (num_lines + n_parts - 1) // n_parts
        line_ids = list()
        for i in range(n_parts):
            start = lines_per_part * i
            end = start + lines_per_part
            line_ids.append(indices[start: end])
        for key, take_id in zip(to_split_keys, take):
            action = Split()
            self.sources[key] = action(self.sources[key], line_ids=line_ids)[take_id]

    decompress = _apply_action(Decompress)
    preprocess = _apply_action(Preprocess)
    apply_bpe = _apply_action(ApplyBpe)
    collapse = _apply_action(Collapse)

    def parse(self, *, folder: Path = None):
        for k in self.sources:
            action = Parse(k.lang)
            self.sources[k] = action(self.sources[k], vocab=self.vocab, folder=folder)

    def binarize(self):
        action = Binarize()
        for k in self.sources:
            self.sources[k] = action(self.sources[k], vocab=self.vocab)

    def convert_eat(self, *, folder: Path = None):
        action = ConvertEat()
        for k in self.sources:
            eat_files = action(self.sources[k], folder=folder)
            eat, plain, line_no = eat_files
            self.sources[k] = eat
            self.eat_aux_files[k] = EatAuxFiles(plain, line_no)

    def convert_neo(self, *, folder: Path = None):
        action = ConvertNeo()
        for k in self.sources:
            eat_files = action(self.sources[k], folder=folder)
            neo, plain, line_no = eat_files
            self.sources[k] = plain  # NOTE(j_luo)  Use plain text as the source for bpe later.
            self.neo_aux_files[k] = NeoAuxFiles(neo, line_no)

    def extract_joint_vocab(self, src_key1: Hashable, src_key2: Hashable):
        if self.vocab is not None:
            raise RuntimeError(f'vocab has already been set.')
        action = ExtractJointVocab()
        src1 = self.sources[src_key1]
        src2 = self.sources[src_key2]
        self.vocab = action([src1, src2])

    def link(self, src_key: Hashable, tgt: FormatFile):
        action = Link()
        src = self.sources[src_key]
        action(src, link=tgt)
        self.sources[src_key] = tgt

    def align(self, src_key1: Hashable, src_key2: Hashable, *, op='eat'):
        if op not in ['eat', 'neo']:
            raise ValueError(f'op "{op}" not supported.')
        aux_files = self.eat_aux_files if op == 'eat' else self.neo_aux_files

        src1 = self.sources[src_key1]
        src2 = self.sources[src_key2]
        src_line_no1 = aux_files[src_key1].line_no
        src_line_no2 = aux_files[src_key2].line_no
        action = Align()
        new_file1, new_file2 = action([src1, src2], line_nos=[src_line_no1, src_line_no2])
        self.sources[src_key1] = new_file1
        self.sources[src_key2] = new_file2
