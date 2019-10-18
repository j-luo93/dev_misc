from unittest import TestCase

import torch
import torch.nn as nn

from .named_tensor import collapse, embed


class TestNamedTensor(TestCase):

    def has_name(self, tensor, names):
        self.assertTupleEqual(tensor.names, names)

    def test_embed(self):
        tensor = torch.randint(10, (10, 10)).refine_names('batch', 'length')
        emb = nn.Embedding(10, 20)
        tensor = embed(emb, tensor, 'dim')
        self.has_name(tensor, ('batch', 'length', 'dim'))

    def test_collapse(self):
        tensor = torch.randn(10, 10, 10, names=('a', 'b', 'c'))
        tensor1 = collapse(tensor, 'b', 'c')
        self.has_name(tensor1, ('a', 'bXc'))
