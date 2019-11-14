from unittest import TestCase

from .name import Name


class TestName(TestCase):

    def test_basic_snake(self):
        name = Name('snake', ['this', 'is', 'a', 'test'])
        self.assertEqual(name.value, 'this_is_a_test')

    def test_basic_Camel(self):
        name = Name('camel', ['this', 'is', 'a', 'test'])
        self.assertEqual(name.value, 'ThisIsATest')

    def test_lowercase(self):
        name = Name('snake', ['This', 'is', 'a', 'BIG', 'test'])
        new_name = name.lowercase
        self.assertEqual(new_name.value, 'this_is_a_big_test')

        name = Name('camel', ['This', 'is', 'a', 'BIG', 'test'])
        with self.assertRaises(RuntimeError):
            name.lowercase

    def test_uppercase(self):
        name = Name('snake', ['This', 'is', 'a', 'BIG', 'test'])
        new_name = name.uppercase
        self.assertEqual(new_name.value, 'THIS_IS_A_BIG_TEST')

        name = Name('camel', ['This', 'is', 'a', 'BIG', 'test'])
        with self.assertRaises(RuntimeError):
            name.uppercase

    def test_conversion(self):
        name = Name('snake', ['This', 'is', 'a', 'BIG', 'test'])
        self.assertEqual(name.value, 'This_is_a_BIG_test')
        name = name.camel
        self.assertEqual(name.value, 'ThisIsABIGTest')

    def test_format(self):
        name = Name('snake', ['{name}', 'is', 'a', 'test'])
        self.assertEqual(name.format(name='That').value, 'That_is_a_test')
