from unittest import TestCase

from .tracker import Tracker, clear_stages


class TestTracker(TestCase):

    def setUp(self):
        clear_stages()
        self.tracker = Tracker('main')
        stage = self.tracker.add_stage('round', 10)
        self.first_stage = stage.add_stage('first', 200)
        stage.add_stage('second', 300)
        self.tracker.fix_schedule()

    def test_basic(self):
        cnt = 0
        while not self.tracker.finished:
            cnt += 1
            self.tracker.update()
        self.assertEqual(cnt, 5000)
