import sys
from unittest import TestCase
from unittest.mock import MagicMock

from . import parser
from .argument import FormatError


class TestParser(TestCase):
    
    def setUp(self):
        parser.clear_parser()
        sys.argv = ['dummy.py']

    def test_duplicates(self):
        parser.add_argument('--option1', '-o1')
        with self.assertRaises(parser.DuplicateError):
            parser.add_argument('--option1')
        with self.assertRaises(parser.DuplicateError):
            parser.add_argument('--option2', '-o1')

    def test_format(self):
        with self.assertRaises(FormatError):
            parser.add_argument('-option1')
        with self.assertRaises(FormatError):
            parser.add_argument('--option1', '--o1')
        with self.assertRaises(FormatError):
            parser.add_argument('option1')
        with self.assertRaises(FormatError):
            parser.add_argument('--option1', 'o1')
        
    def test_match(self):
        parser.add_argument('--option1', '-o1')
        parser.add_argument('--option2', '-o2')
        parser.add_argument('--option3', '-o3')
        parser.add_argument('--long_option4', '-lo4')
        a = parser.get_argument('--option1')
        a = parser.get_argument('-o2')
        a = parser.get_argument('--l')
        a = parser.get_argument('-l')
        with self.assertRaises(parser.MultipleMatchError):
            parser.get_argument('--option')
        with self.assertRaises(parser.MultipleMatchError):
            parser.get_argument('-o')
    
    def test_value(self):
        a = parser.add_argument('--option', '-o', default=1, dtype=int)
        self.assertEqual(a, 1)
        a = 2
        self.assertEqual(a, 2)
    
    def test_parse_args(self):
        parser.add_argument('--option1', '-o1', default=1, dtype=int)
        parser.add_argument('--option2', default='test')
        sys.argv = 'dummy.py -o1 2'.split()
        parser.parse_args()
        a = parser.get_argument('-o1')
        self.assertEqual(a, 2)

    def _get_cfg_mock(self):
        reg = MagicMock()
        def side_effect():
            class Test:
                def __init__(self):
                    self.x = 1
            return Test()

        reg.__getitem__.return_value = MagicMock(side_effect=side_effect)
        parser.add_cfg_registry(reg)

    def test_cfg_default(self):
        self._get_cfg_mock()
        parser.add_argument('--x', default=0, dtype=int)
        sys.argv = 'dummy.py -cfg Test'.split()
        parser.parse_args()
        a = parser.get_argument('x')
        self.assertEqual(a, 1)

    def test_cfg_cli(self):
        self._get_cfg_mock()
        parser.add_argument('--x', default=0, dtype=int)
        sys.argv = 'dummy.py -cfg Test --x 2'.split()
        parser.parse_args()
        a = parser.get_argument('x')
        self.assertEqual(a, 2)
    
    def test_keywords(self):
        with self.assertRaises(parser.KeywordError):
            parser.add_argument('--help')
        with self.assertRaises(parser.KeywordError):
            parser.add_argument('-h')
        with self.assertRaises(parser.KeywordError):
            parser.add_argument('--unsafe')
        with self.assertRaises(parser.KeywordError):
            parser.add_argument('-u')
    
    def _get_test_obj(self):
        class Test:
            def __init__(self):
                self.y = parser.get_argument('y', default=2)
        return Test()

    def test_ad_hoc_in_class(self):
        parser.add_argument('--x', default=1, dtype=int)
        sys.argv = 'dummy.py -u'.split()
        parser.parse_args()
        obj = self._get_test_obj()
        self.assertEqual(obj.y, 2)

    def test_ad_hoc_in_class_overridden_by_cli(self):
        parser.add_argument('--x', default=1, dtype=int)
        sys.argv = 'dummy.py -u --y 3'.split()
        parser.parse_args()
        obj = self._get_test_obj()
        self.assertEqual(obj.y, 3)

    def test_ad_hoc_in_cli(self):
        parser.add_argument('--x', default=1)
        sys.argv = 'dummy.py --y 2 -u'.split()
        parser.parse_args()
        a = parser.get_argument('y')
        self.assertEqual(a, 2)