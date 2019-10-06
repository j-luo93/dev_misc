from unittest import TestCase

import pandas as pd
import torch

from .pandas import PandasDataLoader, PandasDataset


def _set_up_basic_df():
    df = pd.DataFrame()
    df['x'] = range(10)
    df['y'] = range(10, 20)
    return df


class TestPandasDataset(TestCase):

    def test_basic(self):
        df = _set_up_basic_df()
        ds = PandasDataset(df)
        self.assertEqual(len(ds), 10)
        self.assertListEqual(ds[5].tolist(), [5, 15])

    def test_columns(self):
        df = _set_up_basic_df()
        ds = PandasDataset(df, columns=['x'])
        self.assertListEqual(ds[5].tolist(), [5])


class TestPandasDataLoader(TestCase):

    def test_iter(self):
        df = _set_up_basic_df()
        df['y'] = df['y'].astype('float32')
        dataloader = PandasDataLoader(df, batch_size=2)
        cnt = 0
        for batch in dataloader:
            cnt += 1
            self.assertEqual(batch.x.dtype, torch.int64)
            self.assertEqual(batch.y.dtype, torch.float32)
        self.assertEqual(cnt, 5)
