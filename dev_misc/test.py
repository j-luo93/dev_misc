from unittest import TestCase as TC

import numpy as np
import torch

from dev_misc.arglib import reset_repo
from dev_misc.devlib import NDA
from dev_misc.devlib.named_tensor import (patch_named_tensors,
                                          unpatch_named_tensors)
from dev_misc.devlib.tensor_x import TensorX


def _to_array(array_like):
    if torch.is_tensor(array_like):
        ret = array_like.cpu().numpy()
    elif isinstance(array_like, TensorX):
        ret = array_like.cpu().numpy()
    else:
        ret = np.asarray(array_like)
    return ret


class TestCase(TC):

    def setUp(self):
        reset_repo()
        patch_named_tensors()

    def tearDown(self):
        unpatch_named_tensors()

    def assertArrayAlmostEqual(self, first, second, *args, **kwargs):
        first = _to_array(first)
        second = _to_array(second)
        np.testing.assert_almost_equal(first, second, *args, **kwargs)
