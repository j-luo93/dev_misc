from unittest import TestCase

from .argument import (Argument, DtypeNotAllowed, NameFormatError,
                       NArgsNotAllowed)


class TestArgument(TestCase):

    def test_default_value(self):
        arg = Argument('option', dtype=int)
        self.assertEqual(arg.value, None)
        arg = Argument('option', dtype=int, default=0.1)
        self.assertEqual(arg.value, 0)
        arg = Argument('option', dtype=int, default=1)
        self.assertEqual(arg.value, 1)

    def test_dtypes(self):
        arg = Argument('option', dtype=int, default=1)
        arg = Argument('option', dtype=float, default=1)
        arg = Argument('option', dtype=str, default=1)
        arg = Argument('option', dtype=bool, default=1)
        with self.assertRaises(DtypeNotAllowed):
            arg = Argument('option', dtype=tuple, default=1)

    def test_name_format(self):
        with self.assertRaises(NameFormatError):
            Argument('no_option')

    def test_nargs_format(self):
        Argument('one', nargs=1)
        Argument('second', nargs=2)
        Argument('plus', nargs="+")
        with self.assertRaises(NArgsNotAllowed):
            Argument('wrong', nargs=2.3)
