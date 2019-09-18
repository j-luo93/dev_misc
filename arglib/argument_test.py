from .argument import DTypeNotAllowed, Argument

from unittest import TestCase


class TestArgument(TestCase):

    def test_default_value(self):
        arg = Argument('--option', dtype=int)
        self.assertEqual(arg.value, None)
        arg = Argument('--option', dtype=int, default=0.1)
        self.assertEqual(arg.value, 0)
        arg = Argument('--option', dtype=int, default=1)
        self.assertEqual(arg.value, 1)

    def test_dtypes(self):
        arg = Argument('--option', dtype=int, default=1)
        arg = Argument('--option', dtype=float, default=1)
        arg = Argument('--option', dtype=str, default=1)
        arg = Argument('--option', dtype=bool, default=1)
        with self.assertRaises(DTypeNotAllowed):
            arg = Argument('--option', dtype=tuple, default=1)
