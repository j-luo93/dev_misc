from unittest import TestCase

import torch
import torch.nn as nn
from torch.nn.modules import MultiheadAttention

from .named_tensor import (adv_index, embed, expand_as, gather,
                           get_named_range, leaky_relu, self_attend, patch_named_tensors, unpatch_named_tensors)


class TestNamedTensorBase(TestCase):

    def has_name(self, tensor, names):
        self.assertTupleEqual(tensor.names, names)

    def has_shape(self, tensor, shape):
        self.assertTupleEqual(tensor.shape, shape)


class TestNamedTensorOldHelperFunctions(TestNamedTensorBase):

    def test_embed(self):
        tensor = torch.randint(10, (10, 10)).refine_names('batch', 'length')
        emb = nn.Embedding(10, 20)
        tensor = embed(emb, tensor, 'emb')
        self.has_name(tensor, ('batch', 'length', 'emb'))

    def test_self_attend(self):
        mod = MultiheadAttention(40, 8)
        tensor = torch.randn(13, 32, 40, names=['length', 'batch', 'repr'])
        output, weight = self_attend(mod, tensor, 'self_attn_repr')
        self.has_name(output, ('length', 'batch', 'self_attn_repr'))
        self.has_name(weight, ('batch', 'length', 'length_T'))

    def test_adv_index(self):
        tensor = torch.randn(32, 10, 10, names=['x', 'y', 'z'])
        index = torch.randint(10, (3, )).refine_names('w')
        tensor = adv_index(tensor, 'z', index)
        self.has_name(tensor, ('x', 'y', 'w'))
        self.has_shape(tensor, (32, 10, 3))

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

    def test_expand_as(self):
        tensor = torch.randn(32, names=['batch'])
        other = torch.randn(32, 10, names=['batch', 'repr'])
        ret = expand_as(tensor, other)
        self.has_name(ret, ('batch', 'repr'))
        self.has_shape(ret, (32, 10))

    def test_get_named_range(self):
        ret = get_named_range(32, 'batch')
        self.has_name(ret, ('batch', ))
        self.has_shape(ret, (32, ))


class TestNamedTensorPath(TestNamedTensorBase):

    def setUp(self):
        patch_named_tensors()

    def tearDown(self):
        unpatch_named_tensors()

    def test_leaky_relu(self):
        tensor = torch.randn(32, 10, names=['batch', 'repr'])
        tensor = leaky_relu(tensor)
        self.has_name(tensor, ('batch', 'repr'))
