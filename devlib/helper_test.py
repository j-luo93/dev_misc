from dataclasses import dataclass
from unittest import TestCase

import numpy as np
import torch

from .helper import dataclass_size_repr, get_range, get_tensor


def _to_list(tensor):
    return tensor.cpu().numpy().tolist()


class TestHelper(TestCase):

    def test_get_tensor(self):
        x = np.random.randn(10)
        t = get_tensor(x)
        self.assertListEqual(_to_list(t), x.tolist())

        new_t = get_tensor(t)
        self.assertListEqual(_to_list(new_t), t.cpu().numpy().tolist())

    def test_get_range(self):
        r = get_range(10, 2, 1)
        self.assertListEqual(_to_list(r), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    def test_dataclass_size_repr(self):

        @dataclass
        class Test:
            x: int
            y: torch.Tensor
            z: np.ndarray

            __repr__ = dataclass_size_repr

        test = Test(1, torch.randn(32, 10), np.zeros([2, 10]))
        ans = 'Test(x=1, y: (32, 10), z: (2, 10))'
        self.assertEqual(repr(test), ans)
