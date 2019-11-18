import random
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, Hashable, List, Union, Optional

from deprecated import deprecated

from .action import (Align, ApplyBpe, Binarize, Collapse, ConvertEat,
                     ConvertNeo, Decompress, Download, ExtractJointVocab, Link,
                     Merge, Parse, Preprocess, Split)
from .format_file import FormatFile


@dataclass
class EatNeoFiles:
    main: FormatFile
    plain: FormatFile
    line_no: Optional[FormatFile] = None


Sources = Dict[Hashable, Union[FormatFile, str, EatNeoFiles]]


def _apply_action(action_cls):

    @wraps(action_cls.__call__)
    def wrapper(self, **kwargs):
        action = action_cls()
        for k in self.sources:
            if isinstance(self.sources[k], (str, FormatFile)):
                self.sources[k] = action(self.sources[k], **kwargs)
            else:
                src = self.sources[k]
                main = action(src.main, **kwargs)
                plain = action(src.plain, **kwargs)
                self.sources[k] = EatNeoFiles(main, plain, src.line_no)

    return wrapper


class Pipeline:

    def __init__(self, sources: Sources):
        self.sources = sources
        self.vocab = None
        # self.eat_neo_files = dict()

    def download(self, **common_info):
        download_to = dict()
        for k in self.sources:
            info = deepcopy(common_info)
            info.update(k.__dict__)
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
            # HACK(j_luo) Maybe there is way to combine this with _apply_action?
            src = self.sources[k]
            if isinstance(src, (FormatFile, str)):
                self.sources[k] = action(src, vocab=self.vocab)
            else:
                main = action(src.main, vocab=self.vocab)
                plain = action(src.plain, vocab=self.vocab)
                self.sources[k] = EatNeoFiles(main, plain, src.line_no)

    def convert_eat(self, *, graph: bool = False, folder: Path = None):
        action = ConvertEat(graph=graph)
        for k in self.sources:
            eat_files = action(self.sources[k], folder=folder)
            eat, plain, line_no = eat_files
            # self.sources[k] = plain if graph else eat
            # self.eat_neo_files[k] = EatNeoFiles(eat, plain, line_no)
            self.sources[k] = EatNeoFiles(eat, plain, line_no)

    def convert_neo(self, *, linear: bool = False, folder: Path = None):
        action = ConvertNeo(linear=linear)
        for k in self.sources:
            eat_files = action(self.sources[k], folder=folder)
            neo, plain, line_no = eat_files
            # self.sources[k] = neo if linear else plain  # NOTE(j_luo)  Use plain text as the source for bpe later.
            # self.eat_neo_files[k] = EatNeoFiles(neo, plain, line_no)
            # TODO(j_luo) write a class like FormatFiles that can be a collection of files. But doesn't self.sources already do that?
            self.sources[k] = EatNeoFiles(neo, plain, line_no)

    def extract_joint_vocab(self, *src_keys: Hashable):
        if self.vocab is not None:
            raise RuntimeError(f'vocab has already been set.')
        action = ExtractJointVocab()
        srcs = list()
        for src_key in src_keys:
            src = self.sources[src_key]
            if isinstance(src, (FormatFile, str)):
                srcs.append(src)
            else:
                srcs.extend([src.main, src.plain])
        self.vocab = action(srcs)

    def link(self, src_key: Hashable, tgt: FormatFile):
        action = Link()
        src = self.sources[src_key]
        if isinstance(src, (str, FormatFile)):
            action(src, link=tgt)
            self.sources[src_key] = tgt
        else:
            main = src.main
            plain = src.plain
            action(plain, link=tgt)
            if 'eat' in main.fmt.ops:
                main_tgt = tgt.add_op('eat')
            elif 'neo' in main.fmt.ops:
                main_tgt = tgt.add_op('neo')
            else:
                raise ValueError('Not sure why you got here.')
            action(main, link=main_tgt)

    def align(self, src_key1: Hashable, src_key2: Hashable, *, op='eat'):
        if op not in ['eat', 'neo']:
            raise ValueError(f'op "{op}" not supported.')

        src_files1 = self.sources[src_key1]
        src_files2 = self.sources[src_key2]
        src_line_no1 = src_files1.line_no
        src_line_no2 = src_files2.line_no
        action = Align()
        src_main1 = src_files1.main
        src_main2 = src_files2.main
        new_main_file1, new_main_file2 = action([src_main1, src_main2], line_nos=[src_line_no1, src_line_no2])
        src_plain1 = src_files1.plain
        src_plain2 = src_files2.plain
        new_plain_file1, new_plain_file2 = action([src_plain1, src_plain2], line_nos=[src_line_no1, src_line_no2])
        self.sources[src_key1] = EatNeoFiles(new_main_file1, new_plain_file1)
        self.sources[src_key2] = EatNeoFiles(new_main_file2, new_plain_file2)
