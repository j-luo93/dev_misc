from unittest import TestCase as TC

import numpy as np
import torch

from dev_misc.arglib import reset_repo
from dev_misc.devlib import NDA
from dev_misc.devlib.named_tensor import (patch_named_tensors,
                                          unpatch_named_tensors)


class TestCase(TC):

    def setUp(self):
        reset_repo()
        patch_named_tensors()

    def tearDown(self):
        unpatch_named_tensors()

    def assertArrayEqual(self, first, second):

        def tolist(array_like):
            if isinstance(array_like, np.ndarray):
                ret = array_like.tolist()
            elif torch.is_tensor(array_like):
                ret = array_like.cpu().numpy().tolist()
            elif isinstance(array_like, tuple):
                ret = list(array_like)
            else:
                ret = array_like
            return ret

        first = tolist(first)
        second = tolist(second)
        self.assertListEqual(first, second)
