from unittest import TestCase

from .trackable import PBarOutOfBound, CountTrackable, reset_all, MaxTrackable, MinTrackable


class TestCountTrackable(TestCase):

    def setUp(self):
        reset_all()

    def test_update(self):
        x = CountTrackable('step', total=10)
        for i in range(10):
            x.update()
            self.assertEqual(x.value, i + 1)

        with self.assertRaises(PBarOutOfBound):
            x.update()

    def test_update_without_total(self):
        x = CountTrackable('step', total=100)
        for i in range(100):
            x.update()
            self.assertEqual(x.value, i + 1)

    def test_nested(self):
        x = CountTrackable('epoch', total=10)
        y = x.add_trackable('step', total=10)
        for i in range(10):
            for j in range(10):
                y.update()
                self.assertEqual(y.value, j + 1)
            x.update()
            self.assertEqual(x.value, i + 1)

    def test_reset(self):
        x = CountTrackable('epoch', total=10)
        for i in range(5):
            x.update()
        x.reset()
        self.assertEqual(x.value, 0)


class TestMaxMinTrackable(TestCase):

    def setUp(self):
        reset_all()

    def test_update(self):
        x = MaxTrackable('score')
        for i in range(10):
            x.update(i)
        self.assertEqual(x.value, 9)

    def test_reset(self):
        x = MaxTrackable('score')
        for i in range(5):
            x.update(i)
        x.reset()
        self.assertEqual(x.value, -float('inf'))

    def test_min(self):
        x = MinTrackable('loss')
        for i in range(5):
            x.update(i)
        self.assertEqual(x.value, 0)
