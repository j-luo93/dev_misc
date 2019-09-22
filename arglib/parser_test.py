import sys
from unittest import TestCase

from .parser import (DuplicateArgument, MatchNotFound, MultipleMatches,
                     add_argument, g, parse_args, reset_repo)


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
        sys.argv = 'dummy.py --use'.split()
        with self.assertRaises(MultipleMatches):
            parse_args()

    def test_no_match(self):
        add_argument('option', default=1, dtype=int)
        _parse('--option 2')
        self.assertEqual(g.option, 2)

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
