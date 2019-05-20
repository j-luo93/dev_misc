import logging

import numpy as np
import torch

from collections import defaultdict

class Metric:
    
    def __init__(self, name, value, weight, report_mean=True):
        self.name = name
        self._v = value
        self._w = weight
        self._report_mean = report_mean
    
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f'{self._v}/{self._w}={self.mean:.3f}'
    
    def __eq__(self, other):
        return self.name == other.name

    def __add__(self, other):
        if isinstance(other, Metric):
            assert self == other, 'Cannot add two different metrics.'
            assert self.report_mean == other.report_mean
            return Metric(self.name, self._v + other._v, self._w + other._w, report_mean=self.report_mean)
        else:
            # NOTE This is useful for sum() call. 
            assert isinstance(other, (int, float)) and other == 0
            return self
        
    def __radd__(self, other):
        return self.__add__(other)

    @property
    def report_mean(self):
        return self._report_mean

    @property
    def mean(self):
        return self._v / self._w
    
    @property
    def total(self):
        return self._v

class Tracker:

    def __init__(self, metadata=None):
        self.clear_best()
        self.clear()
        self._metadata = metadata

    def clear(self):
        self._metrics = defaultdict(float)

    def clear_best(self):
        self.best_score = None
        self.best_epoch = None

    def update(self, metric):
        self._metrics[metric.name] += metric

    def copy_(self, other):
        assert type(self._metadata) == type(other._metadata)
        self._metadata = other._metadata
        self.best_score = other.best_score
        self.best_epoch = other.best_epoch
        self._metrics = other._metrics

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            try:
                metadata = super().__getattribute__('_metadata')
                return getattr(metadata, name)
            except AttributeError as e:
                raise e 
    
    def set_metadata(self, name, value):
        if not hasattr(self._metadata, name):
            raise AttributeError('No such attribute in metadata')
        setattr(self._metadata, name, value)

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
        for name, metric in self._metrics.items():
            score = metric.mean if metric.report_mean else metric.total
            logging.info('  %s:\t%.2f' %(name, score))
        ret = self._metrics['loss'].mean
        self.clear()
        return ret
