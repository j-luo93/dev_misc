from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from devlib.helper import get_tensor, pad_to_dense


class PandasDataset(Dataset):

    def __init__(self, data: Any, columns: List[str] = None):
        # NOTE(j_luo) The most efficient way to get the n-th example is to access a dataframe by column. Therefore data is transposed first.
        if columns is not None:
            data = data[columns]

        # NOTE(j_luo) There are two copies of data in order to support `select` method.
        self._orig_data_t = data
        self._orig_dtypes = tuple(data.dtypes.tolist())
        self._orig_data = data.T
        self._orig_columns = tuple(data.columns.tolist())
        self._set_attributes(data)

    def _set_attributes(self, data: pd.DataFrame):
        self._data_t = data
        self._dtypes = tuple(data.dtypes.tolist())
        self._data = data.T
        self._columns = tuple(data.columns.tolist())

    @property
    def data(self):
        return self._orig_data_t

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def columns(self):
        return self._columns

    def __len__(self):
        return len(self._data_t)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def select(self, mask: pd.DataFrame):
        data = self._orig_data_t[mask].reset_index(drop=True)
        self._set_attributes(data)


def pandas_collate_fn(batch: List[pd.Series]) -> pd.DataFrame:
    return pd.concat(batch, axis=1).T


class PandasDataLoader(DataLoader):

    def __init__(self, data, *args, columns: List[str] = None, collate_fn: Callable[[List[pd.Series]], Any] = None, **kwargs):
        dataset = PandasDataset(data, columns=columns)
        if collate_fn is None:
            collate_fn = pandas_collate_fn
        super().__init__(dataset, *args, collate_fn=collate_fn, **kwargs)
