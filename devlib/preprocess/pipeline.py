from pathlib import Path
from .action import Link
from copy import deepcopy
from typing import Dict, List, NamedTuple, Union

from .action import (ApplyBpe, Binarize, Decompress, Download,
                     ExtractJointVocab, Merge, Preprocess, Split,
                     set_action_constant)
from .format_file import FormatFile

Sources = Dict[NamedTuple, Union[FormatFile, str]]


class Pipeline:

    def __init__(self, sources: Sources):
        self.sources = sources
        self.vocab_key = None

    def download(self, **common_info):
        download_to = dict()
        for k in self.sources:
            info = deepcopy(common_info)
            info.update(k._asdict())
            fmt_file = FormatFile(**info)
            download_to[k] = fmt_file
        action = Download()
        for k in self.sources:
            action(self.sources[k], download_to=download_to[k])
        self.sources = download_to

    def merge(self, to_merge_keys: List[NamedTuple], merge_to_key: NamedTuple, merge_to: FormatFile):
        to_merge = [self.sources[key] for key in to_merge_keys]
        action = Merge()
        action(to_merge, merge_to=merge_to)
        for key in to_merge_keys:
            del self.sources[key]
        self.sources[merge_to_key] = merge_to

    def split(self, to_split_key: NamedTuple, line_ids: List[List[int]], take: int):
        action = Split()
        self.sources[to_split_key] = action(self.sources[to_split_key], line_ids=line_ids)[take]

    def decompress(self, folder: Path = None):
        action = Decompress()
        for k in self.sources:
            self.sources[k] = action(self.sources[k], folder=folder)

    def preprocess(self, folder: Path = None):
        action = Preprocess()
        for k in self.sources:
            self.sources[k] = action(self.sources[k], folder=folder)

    def apply_bpe(self, codes, folder: Path = None):
        action = ApplyBpe()
        for k in self.sources:
            self.sources[k] = action(self.sources[k], codes=codes, folder=folder)

    def binarize(self, folder: Path = None):
        action = Binarize()
        vocab = self.sources[self.vocab_key]
        for k in self.sources:
            if k.main != 'vocab':
                self.sources[k] = action(self.sources[k], vocab=vocab, folder=folder)

    def extract_joint_vocab(self, src_key: NamedTuple, tgt_key: NamedTuple, vocab_key: NamedTuple):
        if self.vocab_key is not None:
            raise RuntimeError(f'vocab_key has already been set.')
        self.vocab_key = vocab_key
        action = ExtractJointVocab()
        src = self.sources[src_key]
        tgt = self.sources[tgt_key]
        self.sources[vocab_key] = action([src, tgt])

    def link(self, src_key: NamedTuple, tgt: FormatFile):
        action = Link()
        src = self.sources[src_key]
        action(src, link_to=tgt)
        self.sources[src_key] = tgt
