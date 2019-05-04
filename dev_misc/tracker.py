import logging

import numpy as np
import torch

from collections import defaultdict, namedtuple

_Pair = namedtuple('Pair', ['v', 'w'])
class Pair(_Pair):

    def __add__(self, other):
        return Pair(self.v + other.v, self.w + other.w)

    @classmethod
    def merge(cls, *pairs):
        ret = pairs[0]
        for pair in pairs[1:]:
            ret = ret + pair
        return ret

    @property
    def mean(self):
        return self.v / self.w

class Tracker:

    def __init__(self):
        self._epoch = 1
        self.clear_best()
        self.clear()

    def clear(self):
        self._values = defaultdict(float)
        self._weights = defaultdict(float)

    def clear_best(self):
        self.best_score = None
        self.best_epoch = None

    def update(self, name, pair):
        self._values[name] += pair.v
        self._weights[name] += pair.w

    def copy_(self, other):
        self._epoch = other._epoch
        self.best_score = other.best_score
        self.best_epoch = other.best_epoch
        self._values = other._values
        self._weights = other._weights

    @property
    def epoch(self):
        return self._epoch

    def next_epoch(self):
        self._epoch += 1

    def update_best(self, score, mode='min', quiet=False):
        if isinstance(score, torch.Tensor):
            assert score.numel() == 1
            score = score.item()
        elif isinstance(score, np.ndarray):
            assert score.size == 1
            score = score[0]
        updated = False

        def should_update():
            if score is None:
                return False
            if self.best_score is None:
                return True
            if mode == 'max' and self.best_score < score:
                return True
            if mode == 'min' and self.best_score > score:
                return True
            return False

        updated = should_update()
        if updated:
            self.best_score = score
            self.best_epoch = self.epoch
        if self.best_score is not None and not quiet:
            logging.info('Best score is %.3f at epoch %d' %(self.best_score, self.best_epoch))
        return updated

    def output(self):
        logging.info('Epoch %d, summary from tracker:' %self.epoch)
        for name in self._values:
            v = self._values[name]
            w = self._weights[name]
            logging.info('  %s:\t%.2f' %(name, v / w))
        ret = self._values['loss'] / self._weights['loss']
        self.clear()
        return ret
