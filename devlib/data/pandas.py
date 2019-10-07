from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from devlib.helper import get_tensor, pad_to_dense


class PandasDataset(Dataset):

    def __init__(self, data: Any, columns: List[str] = None):
        # NOTE(j_luo) The most efficient way to get the n-th example is to access a dataframe by column. Therefore data is transposed first.
        if columns is not None:
            data = data[columns]
        self._data_t = data
        self._dtypes = tuple(data.dtypes.tolist())
        self._data = data.T
        self._columns = tuple(data.columns.tolist())

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


def pandas_collate_fn(batch: List[pd.Series]) -> pd.DataFrame:
    return pd.concat(batch, axis=1).T


class PandasDataLoader(DataLoader):

    def __init__(self, data, *args, columns: List[str] = None, collate_fn: Callable[[List[pd.Series]], Any] = None, **kwargs):
        dataset = PandasDataset(data, columns=columns)
        if collate_fn is None:
            collate_fn = pandas_collate_fn
        super().__init__(dataset, *args, collate_fn=collate_fn, **kwargs)
