from . import parser
import sys
from unittest import TestCase
from unittest.mock import MagicMock, Mock


class TestParser(TestCase):
    
    def setUp(self):
        parser.clear_parser()

    def test_duplicates(self):
        parser.add_argument('--option1', '-o1')
        with self.assertRaises(parser.DuplicateError):
            parser.add_argument('--option1')
        with self.assertRaises(parser.DuplicateError):
            parser.add_argument('--option2', '-o1')

    def test_format(self):
        with self.assertRaises(ValueError):
            parser.add_argument('-option1')
        with self.assertRaises(ValueError):
            parser.add_argument('--option1', '--o1')
        with self.assertRaises(ValueError):
            parser.add_argument('option1')
        with self.assertRaises(ValueError):
            parser.add_argument('--option1', 'o1')
        
    def test_match(self):
        parser.add_argument('--option1', '-o1')
        parser.add_argument('--option2', '-o2')
        parser.add_argument('--option3', '-o3')
        parser.add_argument('--long_option4', '-lo4')
        a = parser.get_argument('--option1')
        self.assertEqual(a.short_name, '-o1')
        a = parser.get_argument('-o2')
        self.assertEqual(a.full_name, '--option2')
        a = parser.get_argument('--l')
        self.assertEqual(a.full_name, '--long_option4')
        a = parser.get_argument('-l')
        self.assertEqual(a.short_name, '-lo4')
        with self.assertRaises(parser.MultipleMatchError):
            parser.get_argument('--option')
        with self.assertRaises(parser.MultipleMatchError):
            parser.get_argument('-o')
    
    def test_value(self):
        a = parser.add_argument('--option', '-o', default=1, type=int)
        self.assertEqual(a.value, 1)
        a.value = 2
        self.assertEqual(a.value, 2)
    
    def test_parse_args(self):
        parser.add_argument('--option1', '-o1', default=1, type=int)
        parser.add_argument('--option2', default='test')
        sys.argv = 'dummy.py -o1 2'.split()
        parser.parse_args()
        a = parser.get_argument('-o1')
        self.assertEqual(a.value, 2)

    def _get_cfg_mock(self):
        reg = MagicMock()
        def side_effect():
            class Test:
                def __init__(self):
                    self.x = 1
            return Test()

        reg.__getitem__.return_value = Mock(side_effect=side_effect)
        return reg

    def test_cfg_default(self):
        reg = self._get_cfg_mock()
        parser.add_cfg_registry(reg)
        sys.argv = 'dummy.py -cfg Test'.split()
        parser.parse_args()
        a = parser.get_argument('--x')
        self.assertEqual(a.value, 1)

    def test_cfg_cli(self):
        reg = self._get_cfg_mock()
        parser.add_cfg_registry(reg)
        sys.argv = 'dummy.py -cfg Test --x 2'.split()
        parser.parse_args()
        a = parser.get_argument('--x')
        self.assertEqual(a.value, 2)