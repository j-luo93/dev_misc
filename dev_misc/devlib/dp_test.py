import torch

from dev_misc import TestCase, test_with_arguments

from .dp import Fibonacci, Hmm, Lis
from .tensor_x import TensorX as Tx

_BS = 32  # Batch size.


class TestDP(TestCase):

    def test_fibonacci(self):
        a0 = Tx(torch.zeros(_BS).long(), ['batch'])
        a1 = Tx(torch.ones(_BS).long(), ['batch'])
        dp = Fibonacci(a0, a1, 10)
        dp.run()
        ans = torch.full([_BS], 55, dtype=torch.long)
        self.assertArrayAlmostEqual(dp[10].data, ans)

    def test_hmm(self):
        # FIXME(j_luo) extreme cases?
        # FIXME(j_luo) variable length?
        tp = Tx(torch.Tensor([[0.7, 0.3], [0.3, 0.7]]), ['y', "yn"])
        ep = Tx(torch.Tensor([[0.9, 0.1], [0.2, 0.8]]), ['y', 'x'])
        obs = Tx(torch.LongTensor([[0, 0, 1, 0, 0]]), ['batch', 'l'])
        p0 = Tx(torch.Tensor([[0.5, 0.5]]), ['batch', 'y'])
        dp = Hmm(obs, p0, tp, ep)
        dp.run()
        ans0 = [[0.8673, 0.1327]]
        ans1 = [[0.8204, 0.1796]]
        ans2 = [[0.3075, 0.6925]]
        ans3 = [[0.8204, 0.1796]]
        ans4 = [[0.8673, 0.1327]]

        def test(index: int, ans):
            res = dp['pm', index].align_to('batch', 'y').data
            self.assertArrayAlmostEqual(res, ans, decimal=4)

        test(0, ans0)
        test(1, ans1)
        test(2, ans2)
        test(3, ans3)
        test(4, ans4)

    def test_lis(self):
        a = Tx(torch.LongTensor([[2, 4, 3, 5, 2, 1, 2, 7, 9]]), ['batch', 'l'])
        dp = Lis(a)
        dp.run()
        ans = [5]
        self.assertArrayAlmostEqual(dp[9], ans)
