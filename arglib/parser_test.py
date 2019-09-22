from unittest import TestCase
from .parser import add_argument, reset_repo, g


class TestAddArgument(TestCase):

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
