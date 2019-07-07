from unittest import TestCase

from .curriculum_pbar import CurriculumPBar, CurriculumProperty, clear_c_props


class TestCurriculumProperty(TestCase):

    def setUp(self):
        clear_c_props()

        class Test:
            prop = CurriculumProperty('prop')

        self.obj = Test()

    def test_prop(self):
        pbar = CurriculumPBar()

        with self.assertRaises(AttributeError):
            self.obj.prop
        with self.assertRaises(TypeError):
            self.obj.prop = 1

        pbar.add_property('prop')
        pbar.prop = 1
        self.assertEqual(pbar.prop, 1)
        self.assertEqual(self.obj.prop, 1)

    def test_callback(self):
        pbar = CurriculumPBar(total=10)
        pbar.add_property('prop')
        pbar.add_inc_one_callback('prop', 'after')
        pbar.prop = 1
        for _ in range(10):
            pbar.update()
        self.assertEqual(pbar.prop, 2)
