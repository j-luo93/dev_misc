from collections import Counter
from unittest import TestCase

from .trackable import reset_all
from .tracker import BaseSetting, Tracker


class TestTracker(TestCase):

    def setUp(self):
        reset_all()

    def test_basic(self):
        tracker = Tracker()
        tracker.add_trackable('epoch', total=10)
        for i in range(10):
            tracker.update('epoch')
            self.assertEqual(tracker.epoch, i + 1)

    def test_nested(self):
        tracker = Tracker()
        epoch = tracker.add_trackable('epoch', total=10)
        step = epoch.add_trackable('step', total=10)
        for i in range(10):
            for j in range(10):
                tracker.update('step')
                self.assertEqual(tracker.step, j + 1)
            tracker.update('epoch')
            self.assertEqual(tracker.epoch, i + 1)

    def test_settings(self):
        tracker = Tracker()
        tracker.add_trackable('step', total=1000)
        setting1 = BaseSetting('1')
        setting2 = BaseSetting('2')
        tracker.add_settings([setting1, setting2], [1.0, 0.5])
        cnt = Counter()
        for _ in range(1000):
            setting = tracker.draw_setting()
            cnt[setting] += 1
            tracker.update('step')
        self.assertTrue(abs(2 - cnt[setting1] / cnt[setting2]) < 0.5)

    def test_with_count_and_max_trackables(self):
        tracker = Tracker()
        tracker.add_trackable('step', total=100)
        tracker.add_trackable('best', agg_func='max')
        tracker.add_max_trackable('best2')
        cnt = 0
        while not tracker.is_finished('step'):
            tracker.update('step')
            cnt += 2
            tracker.update('best', value=cnt)
            tracker.update('best2', value=cnt * 2)
        self.assertEqual(tracker.step, 100)
        self.assertEqual(tracker.best, 200)
        self.assertEqual(tracker.best2, 400)
