from unittest import TestCase

from .trackable import (CountTrackable, MaxTrackable, MinTrackable,
                        PBarOutOfBound, TrackableRegistry, reset_all)


class TestCountTrackable(TestCase):

    def setUp(self):
        reset_all()

    def test_str(self):
        x = CountTrackable('step', total=10)
        self.assertEqual(str(x), 'step: 0')

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

    def test_endless(self):
        x = CountTrackable('check', endless=True, total=10)
        for i in range(100):
            x.update()


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


class TestTrackableRegistry(TestCase):

    def setUp(self):
        reset_all()
        self.reg = TrackableRegistry()
        self.reg.register_trackable('t1', total=200)
        self.reg.register_trackable('t2', total=100)

    def test_basic(self):
        self.assertEqual(len(self.reg), 2)

    def test_getitem(self):
        self.assertEqual(self.reg['t1'].total, 200)
