from __future__ import annotations

import logging
from functools import wraps
from typing import (Callable, ClassVar, Dict, List, NewType, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import pandas as pd
import torch
from prompt_toolkit import PromptSession, print_formatted_text

from dev_misc.devlib import NDA
from dev_misc.devlib.named_tensor import patch_named_tensors

Tensor = torch.Tensor
# A Name instance can be either one of the named dimensions, or the table itself. In either case,
# they are converted into a normal column in DataFrame.
Name = NewType('Name', str)


# def _convert_to_typed_value(value: str):
#     try:
#         return int(value)
#     except ValueError:
#         try:
#             return float(value)
#         except ValueError:
#             return value


# class _CommandRunner:

#     _runnables: ClassVar[str, Callable] = dict()

#     def __init__(self, inspector: Inspector):
#         self._inspector = inspector

#     @classmethod
#     def runnable(cls, func):

#         name = func.__name__
#         if name in cls._runnables:
#             raise NameError(f'A runnable named {name} already exists.')
#         cls._runnables[name] = func

#     def run(self, input_: str):
#         cmd, *args = input_.split()
#         runnable = self._runnables[cmd]
#         typed_args = [_convert_to_typed_value(arg) for arg in args]
#         runnable(self._inspector, *typed_args)


# _runnable = _CommandRunner.runnable


_evalable = set()


def _can_eval(func_or_name):
    if isinstance(func_or_name, str):
        name = func_or_name
    else:
        name = func_or_name.__name__
    if name in _evalable:
        raise NameError(f'A name {name} already exists.')

    _evalable.add(name)
    return func_or_name


TableLike = Union[Tensor, List, Tuple, np.ndarray]
DF = pd.DataFrame


class Inspector:

    @classmethod
    def _is_evalable(cls, name: str):
        for n in _evalable:
            if name.startswith(n):
                return True
        return False

    def __init__(self):
        try:
            patch_named_tensors()
        except RuntimeError:
            pass

        self._working_table: Tuple[Name, DF] = None
        self._tables: Dict[Name, DF] = dict()

    @property
    def working_table(self) -> DF:
        if self._working_table is None:
            return None
        return self._working_table[1]

    @_can_eval
    def show(self):
        return self.working_table

    def _get_table_from_tensor(self, tensor: Tensor, name: Name) -> DF:
        if any(name is None for name in tensor):
            raise ValueError(f'Tensor must be fully named.')

        indices = list()
        for _name, size in zip(tensor.names, tensor.shape):
            indices.append(torch.arange(size).long().rename(_name).expand_as(tensor).numpy().reshape(-1))
        arr = tensor.detach().cpu().numpy().reshape(-1)
        data = indices + [arr]
        # NOTE(j_luo) For tensors, the name actually corresponds to ids.
        columns = [_name + '_id' for _name in tensor.names] + [name]
        df = pd.DataFrame(zip(*data), columns=columns)
        return df

    def _get_table_from_array(self, arr: NDA, name: Name, dim_names: Optional[Sequence[Name]] = None) -> DF:
        if arr.ndim == 1:
            if dim_names is not None:
                raise TypeError(f'Cannot take dim_names if the table dimenionality is 1.')
            df = pd.DataFrame(arr, columns=[name])

            # For this type of DFs, we need to merge it with any existing table that have matched names.
            for k, t in self._tables.items():
                id_name = name + '_id'
                if id_name in t.columns:
                    new_t = pd.merge(df, t, left_index=True, right_on=id_name)
                    self._tables[k] = new_t
        else:
            if dim_names is None or len(dim_names) != arr.ndim - 1:
                raise TypeError(f'Total number of names do not match.')
            columns = [name] + list(dim_names)
            indices = list()
            for i, size in enumerate(arr.shape):
                tile_shape = arr.shape[:i] + (1,) + arr.shape[i + 1:]
                reshape_shape = [1] * i + [-1] + [i] * (arr.ndim - i - 1)
                indices.append(np.tile(np.arange(size, dtype=np.long).reshape(*reshape_shape), *tile_shape))
            data = indices + [arr]
            df = pd.DataFrame(zip(*data), columns=columns)
        return df

    def add_table(self, table: TableLike, name: Name, dim_names: Optional[Sequence[Name]] = None):
        if name in self._tables:
            raise NameError(f'A table named {name} exists already.')
        if torch.is_tensor(table):
            if dim_names is not None:
                raise TypeError(f'Name the tensor directly instead of passing dim_names.')
            table = self._get_table_from_tensor(table, name)
        else:
            if isinstance(table, (list, tuple)):
                table = np.asarray(table)
            table = self._get_table_from_array(table, name, dim_names=dim_names)

        self._tables[name] = table

    def merge(self, name1: Name, name2: Name) -> DF:
        table1 = self._tables[name1]
        table2 = self._tables[name2]
        table1.merge(table2)

    @_can_eval
    def take(self, name: Name, value: Optional = None):
        if value is None:
            self._working_table = (name, self._tables[name])
            print_formatted_text(f'"{name}" table taken.')
        elif self._working_table is None:
            raise RuntimeError(f'Must take one table first.')
        else:
            df_name, df = self._working_table
            df = df[df[name] == value]
            self._working_table = (df_name, df)
            print_formatted_text(f'"{name}" == {value} in "{df_name}" table taken.')

    @_can_eval
    def reset(self):
        self._working_table = None

    def run(self):
        session = PromptSession()
        while True:
            input_ = session.prompt('>>> What do you want?\n')

            if not input_.startswith('self'):
                if self._is_evalable(input_):
                    input_ = 'self.' + input_
                else:
                    print_formatted_text('>>> Cannot understand the prompt.')
                    continue

            try:
                ret = eval(input_)
                if ret is not None:
                    print_formatted_text(ret)
            except Exception as e:
                print_formatted_text('>>> Cannot evaluate the prompt.')
                logging.exception(e)
                continue
