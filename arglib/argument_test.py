from unittest import TestCase

from .argument import Argument


class TestArgument(TestCase):

    def test_str(self):
        a = Argument('--x')
        self.assertEqual(str(a), '--x')
        a = Argument('--x', '-x')
        self.assertEqual(str(a), '--x -x')
        a = Argument('--x', '-x', default=1, dtype=int)
        self.assertEqual(str(a), '--x -x (int) [DEFAULT = 1]')
        a = Argument('--x', '-x', default=1, dtype=int, help='test')
        self.assertEqual(str(a), '--x -x (int): test [DEFAULT = 1]')