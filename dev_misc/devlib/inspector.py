from __future__ import annotations

from inspect import isfunction, currentframe
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


_TableLike = Union[Tensor, List, Tuple, np.ndarray]


class NamedDataFrame(pd.DataFrame):
    "Based on https://github.com/pandas-dev/pandas/issues/2485#issuecomment-174577149."

    _metadata = ['tname', 'auto_merge']  # `tname` stands for table name.

    def __init__(self, *args, tname: Optional[Name] = None, auto_merge: bool = True, **kwargs):
        """If `auto_merge` is True, then the inspector will attempt to merge it with any newly created index tables."""
        self.tname = tname
        self.auto_merge = auto_merge
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return NamedDataFrame

    def _combine_const(self, other, *args, **kwargs):
        return super()._combine_const(other, *args, **kwargs).__finalize__(self)

    def merge(self, other, *args, **kwargs) -> NamedDataFrame:
        """Override merge method so that tname can be inherited."""
        ret = super().merge(other, *args, **kwargs)
        ret.tname = self.tname
        return ret


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

        self._working_table: NamedDataFrame = None
        self._tables: Dict[Name, NamedDataFrame] = dict()
        self._working_table_history: List[NamedDataFrame] = list()
        self._pt_in_history: int = None
        self._cmd_history: List[str] = list()

    @property
    def working_table(self) -> NamedDataFrame:
        return self._working_table

    @property
    @_can_eval
    def history(self):
        return self._cmd_history

    def add_table(self, table: _TableLike, name: Name, dim_names: Optional[Sequence[Name]] = None, auto_merge: bool = True, is_index: bool = False, is_mask_index: bool = False):
        """Add a table to the inspector.

        If `is_index` is True, we need to call `set_index` on the filled-in index values.
        If `is_mask_index` is True, we need to construct ids only on the True values. If True, this would be used instead of `is_index`.
        """
        # Check duplicate.
        if name in self._tables:
            raise NameError(f'A table named {name} exists already.')
        # Check names and convert to np.ndarray.
        if torch.is_tensor(table):
            if dim_names is not None:
                raise TypeError(f'Name the tensor directly instead of passing dim_names.')
            if any(name is None for name in table):
                raise ValueError(f'Tensor must be fully named.')
            arr = table.detach().cpu().numpy()
            dim_names = list(table.names)
        elif isinstance(table, (list, tuple)):
            arr = np.asarray(table)
        if arr.ndim == 1:
            # 1d arrays provide a natural name for the index column.
            dim_names = [name]
        else:
            dim_names = dim_names or list()
        if len(dim_names) != arr.ndim:
            raise TypeError(f'Total number of names do not match.')

        # Fill in all the indices.
        df = NamedDataFrame(arr.reshape(-1), columns=[name], tname=name, auto_merge=auto_merge)
        for i, (size, dim_name) in enumerate(zip(arr.shape, dim_names)):
            dim_name = dim_name + '_id'
            tile_shape = arr.shape[:i] + (1,) + arr.shape[i + 1:]
            reshape_shape = [1] * i + [-1] + [1] * (arr.ndim - i - 1)
            index = np.tile(np.arange(size, dtype=np.long).reshape(*reshape_shape), tile_shape)
            df[dim_name] = index.reshape(-1)

        # Set index if needed.
        if is_mask_index or is_index:
            if is_mask_index:
                df = df[df[name]]
                df.reset_index(drop=True, inplace=True)
            else:
                df.set_index([_name + '_id' for _name in dim_names], inplace=True, verify_integrity=True)
            logging.info(f'Index set for table {name!r}.')

            for k, t in self._tables.items():
                id_name = name + '_id'
                if id_name in t.columns and t.auto_merge:
                    logging.info(f'Merging {name!r} table with {t.tname!r} table.')
                    new_t = t.merge(df, how='left', right_index=True, left_on=id_name)
                    self._tables[k] = new_t
                    logging.info(f'{name!r} table with {t.tname!r} table merged.')

        self._tables[name] = df
        logging.info(f'{name!r} table added.')

    def _add_to_history(self):
        self._working_table_history.append(self._working_table)

    @_can_eval
    def show(self, name: Optional[Name] = None):
        if name is None:
            return self.working_table
        else:
            return self._tables[name]

    @_can_eval
    def take(self, name: Name, value: Optional = None):
        if value is None:
            self._working_table = self._tables[name]
            print_formatted_text(f'"{name}" table taken.')
        elif self._working_table is None:
            raise RuntimeError(f'Must take one table first.')
        else:
            df = self._working_table
            df = df[df[name] == value]
            self._working_table = df
            print_formatted_text(f'"{name}" == {value!r} in "{df.tname}" table taken.')
        self._add_to_history()
        return self._working_table

    @_can_eval
    def reset(self):
        self._working_table = None

    @_can_eval
    def undo(self):
        if not self._working_table_history or self._pt_in_history == 0:
            print_formatted_text(f'Nothing to undo.')
        else:
            if self._pt_in_history is None:
                self._pt_in_history = len(self._working_table_history)
            self._pt_in_history -= 1
            self._working_table = self._working_table_history[self._pt_in_history]
            return self._working_table

    @_can_eval
    def redo(self):
        if self._pt_in_history is None or self._pt_in_history == len(self._working_table_history) - 1:
            print_formatted_text(f'Nothing to redo.')
        else:
            if self._pt_in_history is None:
                self._pt_in_history = -1
            self._pt_in_history += 1
            self._working_table = self._working_table_history[self._pt_in_history]
            return self._working_table_history[self._pt_in_history]

    @_can_eval
    def apply_(self, name: Name, func: Callable, new_name: Optional[Name] = None):
        new_name = new_name or name
        self._working_table[name] = self._get_applied(name, func)
        return self._working_table

    def _get_applied(self, name: name, func: Callable):
        return self._working_table[name].apply(func)

    @_can_eval
    def apply(self, name: Name, func: Callable, new_name: Optional[Name] = None):
        new_name = new_name or name
        ret = self._working_table.copy()
        ret[name] = self._get_applied(name, func)
        return ret

    @_can_eval
    def show_all(self):
        for t in self._tables.values():
            print_formatted_text(f'{t.tname.upper():}')
            print_formatted_text(t.describe())

    @_can_eval
    def merge(self, table: Union[table, Name], *args, **kwargs):
        if isinstance(table, str):
            table = self._tables[table]
        self._working_table = self._working_table.merge(table, *args, **kwargs)
        return self._working_table

    @_can_eval
    def pivot(self, *args, **kwargs):
        self._working_table = self._working_table.pivot(*args, **kwargs)
        return self._working_table

    @_can_eval
    def narrow(self, names: Sequence[Name]):
        self._working_table = self._working_table[names]
        return self._working_table

    def run(self):
        session = PromptSession()
        # Inject some useful variable into the local namespace.
        wt = self._working_table
        for tname, table in self._tables.items():
            locals()[tname] = table
        while True:
            raw_input_ = session.prompt('>>> What do you want?\n')

            if raw_input_ in ['quit', 'q', 'exit']:
                break

            input_ = raw_input_
            if not raw_input_.startswith('self'):
                if self._is_evalable(raw_input_):
                    input_ = 'self.' + raw_input_

            try:
                ret = eval(input_)
                # NOTE(j_luo) Call again if the return is a function.
                if isfunction(ret):
                    input_ += '()'
                    ret = eval(input_)
                if ret is not None:
                    print(ret)
            except Exception as e:
                print_formatted_text(f'>>> Cannot evaluate the expression {input_}.')
                logging.exception(e)
                continue

            wt = self._working_table
            # NOTE(j_luo) Only add to history when the command can run.
            self._cmd_history.append(raw_input_)
