import sys
from unittest import TestCase

from .argument import MismatchedNArgs
from .parser import (DuplicateArgument, MatchNotFound, MultipleMatches,
                     OverlappingRegistries, ReservedNameError, add_argument,
                     add_registry, g, init_g_attr, parse_args, reset_repo)
from .registry import Registry


def _parse(argv_str):
    sys.argv = ('dummy.py ' + argv_str).split()
    parse_args()


class TestParser(TestCase):

    def setUp(self):
        reset_repo()

    def test_open(self):
        add_argument('option', default=1, dtype=int)
        self.assertEqual(g.option, 1)

    def test_in_class(self):
        class Test:
            add_argument('option', default=2, dtype=int)
        self.assertEqual(g.option, 2)

    def test_groups(self):
        add_argument('option1')
        add_argument('option2')

        class Group1:
            add_argument('option3')
            add_argument('option4')

        class Group2:
            add_argument('option5')
            add_argument('option6')

        default_args = [arg.name for arg in g.groups['default']]
        self.assertListEqual(default_args, ['option1', 'option2'])

        group1_args = [arg.name for arg in g.groups['Group1']]
        self.assertListEqual(group1_args, ['option3', 'option4'])

        group2_args = [arg.name for arg in g.groups['Group2']]
        self.assertListEqual(group2_args, ['option5', 'option6'])

    def test_duplicate(self):
        add_argument('option')
        with self.assertRaises(DuplicateArgument):
            add_argument('option')

    def test_multiple_match(self):
        add_argument('use_first')
        add_argument('use_second')
        with self.assertRaises(MultipleMatches):
            _parse('--use')

    def test_match(self):
        add_argument('option', default=1, dtype=int)
        _parse('--option 2')
        self.assertEqual(g.option, 2)

    def test_no_match(self):
        add_argument('option', default=1, dtype=int)
        with self.assertRaises(MatchNotFound):
            _parse('--opion 2')

    def test_match_order(self):
        add_argument('option', default=1, dtype=int)
        _parse('--option 2 --option 3')
        self.assertEqual(g.option, 3)

    def test_fuzzy_match(self):
        add_argument('option', default=1, dtype=int)
        _parse('--o 2')
        self.assertEqual(g.option, 2)

    def test_bool_positive(self):
        add_argument('use_this', default=False, dtype=bool)
        _parse('--use_this')
        self.assertEqual(g.use_this, True)

    def test_bool_negative(self):
        add_argument('use_this', default=True, dtype=bool)
        _parse('--no_use_this')
        self.assertEqual(g.use_this, False)

    def test_bool_order(self):
        add_argument('use_this', default=True, dtype=bool)
        _parse('--no_use_this --use_this --no_use_this')
        self.assertEqual(g.use_this, False)

    def test_nargs_plus(self):
        add_argument('option', dtype=int, nargs='+')
        _parse('--option 1 2 3 4 5')
        self.assertTupleEqual(g.option, (1, 2, 3, 4, 5))

    def test_nargs(self):
        add_argument('one', dtype=int, nargs=1)
        add_argument('two', dtype=int, nargs=2)
        _parse('--one 5 --two 4 8')
        self.assertEqual(g.one, 5)
        self.assertTupleEqual(g.two, (4, 8))

    def test_nargs_raise(self):
        add_argument('two', dtype=int, nargs=2)
        with self.assertRaises(MismatchedNArgs):
            _parse('--two 1')

    def test_reserved_names(self):
        with self.assertRaises(ReservedNameError):
            add_argument('keys')

    def _set_up_init_g_attr_default(self, default='property'):

        @init_g_attr(default=default)
        class Test:

            add_argument('option', default=1, dtype=int)
            add_argument('second_option', default=1, dtype=int)

            def __init__(self, arg, option=2, another_option=3):
                pass

        _parse('--option 4')
        return Test(1)

    def test_init_g_attr_property(self):
        x = self._set_up_init_g_attr_default('property')
        self.assertEqual(x.arg, 1)
        self.assertEqual(x.another_option, 3)
        self.assertEqual(x.option, 4)
        with self.assertRaises(AttributeError):
            x.second_option

    def test_init_g_attr_attibute(self):
        x = self._set_up_init_g_attr_default('attribute')
        self.assertEqual(x.arg, 1)
        self.assertEqual(x.another_option, 3)
        self.assertEqual(x.option, 4)
        with self.assertRaises(AttributeError):
            x.second_option

    def test_init_g_attr_none(self):
        x = self._set_up_init_g_attr_default('none')
        with self.assertRaises(AttributeError):
            x.arg
        with self.assertRaises(AttributeError):
            x.another_option
        with self.assertRaises(AttributeError):
            x.option
        with self.assertRaises(AttributeError):
            x.second_option

    def test_init_g_attr_annotations(self):

        @init_g_attr
        class Test:

            add_argument('option', default=1, dtype=int)
            add_argument('second_option', default=1, dtype=int)

            def __init__(self,
                         arg: 'a',
                         option=2,
                         another_option: 'n' = 3):
                pass
        _parse('--option 4')
        x = Test(1)

        self.assertEqual(x.arg, 1)
        self.assertEqual(x.option, 4)
        self.assertEqual(x._option, 4)
        with self.assertRaises(AttributeError):
            x.another_option
        with self.assertRaises(AttributeError):
            x._arg

    def test_add_registry(self):
        reg = Registry('test')
        add_registry('config', reg)
        add_argument('x', default=0, dtype=int)

        @reg
        class Test:
            x: int = 1

        _parse('--config Test')
        self.assertEqual(g.x, 1)

    def _set_up_multiple_registires(self):
        reg1 = Registry('first')
        reg2 = Registry('second')
        add_registry('first_config', reg1)
        add_registry('second_config', reg2)
        add_argument('x', dtype=int, default=0)
        add_argument('y', dtype=int, default=0)
        return reg1, reg2

    def test_add_multiple_registries(self):
        reg1, reg2 = self._set_up_multiple_registires()

        @reg1
        class Test1:
            x: int = 2

        @reg2
        class Test2:
            y: int = 3

        _parse('--first_config Test1 --second_config Test2')
        self.assertEqual(g.x, 2)
        self.assertEqual(g.y, 3)

    def test_overlapping_registries(self):
        reg1, reg2 = self._set_up_multiple_registires()

        @reg1
        class Test1:
            x: int = 2

        @reg2
        class Test2:
            x: int = 3

        with self.assertRaises(OverlappingRegistries):
            _parse('--first_config Test1 --second_config Test2')

    def test_cli_overriding_registry(self):
        reg = Registry('Test')
        add_registry('config', reg)
        add_argument('x', default=0, dtype=int)

        @reg
        class Test:
            x: int = 2

        _parse('--x 5 --config Test ')
        self.assertEqual(g.x, 5)

    def test_hyphen_in_path(self):
        add_argument('first')
        add_argument('second')

        _parse('--first 1 --second sth-with-hyphen/another-with-hyphen')
        self.assertEqual(g.second, 'sth-with-hyphen/another-with-hyphen')

    def test_equal_sign(self):
        add_argument('first', dtype=int)
        add_argument('second', dtype=int)
        _parse('--first=1 --second 2')
        self.assertEqual(g.first, 1)
        self.assertEqual(g.second, 2)
