from collections import Counter
from unittest import TestCase

from .trackable import reset_all
from .tracker import Task, Tracker


class TestTracker(TestCase):

    def setUp(self):
        reset_all()

    def test_basic(self):
        tracker = Tracker()
        tracker.add_trackable('epoch', total=10)
        tracker.ready()
        for i in range(10):
            tracker.update('epoch')
            self.assertEqual(tracker.epoch, i + 1)

    def test_nested(self):
        tracker = Tracker()
        epoch = tracker.add_trackable('epoch', total=10)
        step = epoch.add_trackable('step', total=10)
        tracker.ready()
        for i in range(10):
            for j in range(10):
                tracker.update('step')
                self.assertEqual(tracker.step, j + 1)
            tracker.update('epoch')
            self.assertEqual(tracker.epoch, i + 1)

    def test_tasks(self):
        tracker = Tracker()
        tracker.add_trackable('step', total=1000)
        tracker.ready()
        task1 = Task()
        task2 = Task()
        tracker.add_tasks([task1, task2], [1.0, 0.5])
        cnt = Counter()
        for _ in range(1000):
            task = tracker.draw_task()
            cnt[task] += 1
            tracker.update('step')
        self.assertTrue(abs(2 - cnt[task1] / cnt[task2]) < 0.5)
