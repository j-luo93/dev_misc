'''
Modififed TestCase to handle matrices.
'''
import unittest

import numpy as np
import torch

class TestCase(unittest.TestCase):

    def assertMatrixShapeEqual(self, m1, m2):
        self.assertTupleEqual(m1.shape, m2.shape)

    def assertMatrixEqual(self, m1, m2):
        self.assertMatrixShapeEqual(m1, m2)
        if torch.is_tensor(m1):
            m1 = m1.cpu().numpy()
        if torch.is_tensor(m2):
            m2 = m2.cpu().numpy()
        np.testing.assert_array_almost_equal(m1, m2)

    def assertProbs(self, probs):
        self.assertMatrixEqual(probs.sum(dim=-1).detach(), torch.ones(*probs.shape[:-1]))

main = unittest.main
