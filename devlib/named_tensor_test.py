from unittest import TestCase

import torch
import torch.nn as nn
from torch.nn.modules import MultiheadAttention

from .helper import get_range
from .named_tensor import adv_index, embed, self_attend, leaky_relu, gather


class TestNamedTensor(TestCase):

    def has_name(self, tensor, names):
        self.assertTupleEqual(tensor.names, names)

    def has_shape(self, tensor, shape):
        self.assertTupleEqual(tensor.shape, shape)

    def test_embed(self):
        tensor = torch.randint(10, (10, 10)).refine_names('batch', 'length')
        emb = nn.Embedding(10, 20)
        tensor = embed(emb, tensor, 'emb')
        self.has_name(tensor, ('batch', 'length', 'emb'))

    def test_self_attend(self):
        mod = MultiheadAttention(40, 8)
        tensor = torch.randn(13, 32, 40, names=['length', 'batch', 'repr'])
        output, weight = self_attend(mod, tensor)
        self.has_name(output, ('length', 'batch', 'repr'))
        self.has_name(weight, ('batch', 'length', 'length_T'))

    def test_adv_index(self):
        tensor = torch.randn(32, 10, 10, names=['x', 'y', 'z'])
        index = torch.randint(10, (3, )).refine_names('w')
        tensor = adv_index(tensor, 'z', index)
        self.has_name(tensor, ('x', 'y', 'w'))
        self.has_shape(tensor, (32, 10, 3))

    def test_get_range_with_name(self):
        tensor = get_range(10, 2, 1, name='batch')
        self.has_name(tensor, (None, 'batch'))

    def test_leaky_relu(self):
        tensor = torch.randn(32, 10, names=['batch', 'repr'])
        tensor = leaky_relu(tensor)
        self.has_name(tensor, ('batch', 'repr'))

    def test_gather(self):
        tensor = torch.randn(32, 10, names=['batch', 'length'])
        index1 = torch.randint(10, (32,)).refine_names('batch')
        ret1 = gather(tensor, index1)
        self.has_name(ret1, ('batch', ))
        self.has_shape(ret1, (32, ))

        index2 = torch.randint(32, (10,)).refine_names('length')
        ret2 = gather(tensor, index2)
        self.has_name(ret2, ('length', ))
        self.has_shape(ret2, (10, ))
