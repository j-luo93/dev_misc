from unittest import TestCase

import torch
from .named_tensor import NamedTensor


class TestNamedTensor(TestCase):

    def test_basic(self):
        x = torch.randn(10, 10)
        with self.assertRaises(TypeError):
            NamedTensor(x)
        with self.assertRaises(ValueError):
            NamedTensor(x, names='batch')
        with self.assertRaises(ValueError):
            NamedTensor(x, names=['batch'])
        nt = NamedTensor(x, names=['batch', 'dim'])
        self.assertTupleEqual(nt.names, ('batch', 'dim'))

    def test_inherit(self):
        x = torch.randn(10, 10)
        nt = NamedTensor(x, names=['batch', 'dim'])
        nt_ls = nt.log_softmax(dim=-1)
        nt_add = 1 + nt
        self.assertTupleEqual(nt_ls.names, ('batch', 'dim'))
        self.assertTupleEqual(nt_add.names, ('batch', 'dim'))

    def test_magic_methods(self):
        x = torch.randn(10, 10)
        nt = NamedTensor(x, names=['batch', 'dim'])
        nt = nt + 1
        self.assertTupleEqual(nt.names, ('batch', 'dim'))
