from unittest import TestCase

from .decorator import not_supported_argument_value, try_when
from .parser import reset_repo, set_argument, test_with_arguments


class TestDecorator(TestCase):

    def setUp(self):
        reset_repo()

    def test_not_supported(self):
        test_with_arguments(option=1, _force=True)

        @not_supported_argument_value('option', 1)
        def test(x):
            return x + 1

        with self.assertRaises(NotImplementedError):
            test(1)

        set_argument('option', 2, _force=True)
        self.assertEqual(test(1), 2)

    def test_not_supported_with_default_value_as_None(self):
        test_with_arguments(option='test', _force=True)

        @not_supported_argument_value('option')
        def test(x):
            return x + 1

        with self.assertRaises(NotImplementedError):
            test(1)

    def test_try_when(self):
        test_with_arguments(option=2, _force=True)

        lst = [1]
        @try_when('option', 1)
        def test():
            lst.append(2)

        test()
        self.assertListEqual(lst, [1])

        set_argument('option', 1, _force=True)
        test()
        self.assertListEqual(lst, [1, 2])
