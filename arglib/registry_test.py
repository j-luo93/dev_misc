from unittest import TestCase
from .registry import Registry, DuplicateInRegistry, NotRegistered


class TestRegistry(TestCase):

    def test_register(self):
        reg = Registry('test')

        @reg
        class Test1:
            x: int = 1

        @reg
        class Test2(Test1):
            y: str = 'test'

        self.assertIs(reg['Test1'], Test1)
        self.assertIs(reg['Test2'], Test2)
        cfg1 = Test1()
        cfg2 = Test2()
        self.assertEqual(cfg1.x, 1)
        self.assertEqual(cfg2.y, 'test')

    def test_multiple_registries(self):
        reg1 = Registry('first')
        reg2 = Registry('second')

        @reg1
        class Test1:
            x: int = 1

        @reg2
        class Test2:
            y: str = 'test'

        self.assertIs(reg1['Test1'], Test1)
        self.assertIs(reg2['Test2'], Test2)

    def test_duplicate(self):
        reg = Registry('test')

        @reg
        class Test:
            pass

        with self.assertRaises(DuplicateInRegistry):
            @reg
            class Test:
                pass

    def test_not_registered(self):
        reg = Registry('test')

        @reg
        class Test:
            pass

        with self.assertRaises(NotRegistered):
            reg['Testt']
