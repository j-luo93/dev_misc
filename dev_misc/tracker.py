import logging

import numpy as np
import torch

from .metrics import plain

class Tracker:

    def __init__(self, metadata=None):
        self.clear_best()
        self._metadata = metadata

    def clear_best(self):
        self.best_score = None
        self.best_epoch = None

    def copy_(self, other):
        assert type(self._metadata) == type(other._metadata)
        self._metadata = other._metadata
        self.best_score = other.best_score
        self.best_epoch = other.best_epoch

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
        score = plain(score)
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