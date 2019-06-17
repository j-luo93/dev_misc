import logging
import itertools
import time

import enlighten
import numpy as np
import torch

from .metrics import plain

_manager = enlighten.get_manager()
class _Stage:

    def __init__(self, name, num_steps=1, parent=None):
        # These attributes determine the stage's layout.
        self._name = name
        self._num_steps = num_steps
        self._stages = list()

        # These attributes are needed to track the progress.
        self._step = 0 # NOTE Current step for this Stage object.
        self._current_substage_idx = 0 # NOTE Current substage idx.
        self._current_stage = None # NOTE Current stage (the stage just yielded).
        self._parent = parent
        self._root_cache = None # NOTE Store the root node (to flatten the tree structure).
        self._pbars = dict()
        self._generator = None # NOTE Store the generator for the root node only.
        
        # Add a pbar if num_steps > 1.
        if self._num_steps > 1:
            self.add_pbar(name, total=num_steps)
    
    def load_state_dict(self, state_dict):
        self._step = state_dict['_step']
        self._current_substage_idx = state_dict['_current_substage_idx']
        self._current_stage = state_dict['_current_stage']
        for name, pbar_meta in state_dict['_pbars'].items():
            pbar = self._pbars[name]
            pbar.count = pbar_meta['count']
            pbar.refresh()
        for s1, s2 in zip(self._stages, state_dict['_stages']):
            s1.load_state_dict(s2)
    
    def state_dict(self):
        ret = {'_step': self._step, '_current_substage_idx': self._current_substage_idx, '_current_stage': self._current_stage}
        ret['_pbars'] = {name: {'count': pbar.count} for name, pbar in self._pbars.items()} # NOTE pbar itself cannot be serialized for some reason.
        stage_ret = list()
        for s in self._stages:
            stage_ret.append(s.state_dict())
        ret['_stages'] = stage_ret
        return ret
    
    def _update_pbars(self):
        for pbar in self._pbars.values():
            if pbar.total == pbar.count:
                pbar.count = 0
                pbar.start = time.time()
            pbar.update()

    def _get_stages(self):
        if self._stages: # NOTE This code block deals with non-terminal nodes.

            def safe_next(gen): # Get the next item safely.
                try:
                    item = next(gen)
                except StopIteration:
                    return False, None
                return True, item
                
            while self._step < self._num_steps:  # This stage needs to be called this many times.
                while self._current_substage_idx < len(self._stages):  # We have to go through each sub-stages.
                    child_node = self._stages[self._current_substage_idx]
                    generator = child_node._get_stages()
                    while True:
                        safe, item = safe_next(generator)
                        if not safe:
                            break
                        next_safe, next_item = safe_next(generator) # NOTE This is needed to check if we have reached the last sub-stage.
                        if next_safe:
                            generator = itertools.chain([next_item], generator)
                        else:
                            self._current_substage_idx += 1
                            if self._current_substage_idx == len(self._stages):
                                self._update_pbars() # NOTE This must be called before yielding.
                                self._current_substage_idx = 0
                                self._step += 1
                        yield item
            self._step = 0
        else: # NOTE This deals with terminal nodes.
            logging.info(f'Entering {self}.')
            while self._step < self._num_steps:
                self._update_pbars() # NOTE This must be called before yielding.
                if self._step == self._num_steps - 1:
                    logging.info(f'Ending {self}.')
                snapshot = str(self) # NOTE Already yield a snapshot since attributes like _step is constantly changing due to next_safe check.
                self._step += 1
                yield snapshot
            self._step = 0

    def update(self):
        """Always go back to the stage tracked by the root."""
        root = self.root
        if root._generator is None:
            root._generator = root._get_stages() # NOTE This only need to be done once.
        
        self._current_stage = next(root._generator)

    @property
    def root(self):
        if self._parent is None:
            return self
        elif self._root_cache is None:
            self._root_cache = self._parent.root
        return self._root_cache

    def __repr__(self):
        return f'Stage(name="{self._name}", num_steps={self._num_steps})'
    
    def __str__(self):
        if self._name == '_main' or self._parent is None:
            return ''
        ret = str(self._parent)
        if ret:
            ret += ' -> '
        ret += f'"{self._name}: {self._step + 1}"'
        return ret
    
    def add_stage(self, name, num_steps=1):
        assert all([name != stage._name for stage in self._stages])

        stage = _Stage(name, num_steps=num_steps, parent=self)
        self._stages.append(stage)
        return stage

    def add_pbar(self, name, total=None, unit='samples'):
        if name in self._pbars:
            raise NameError(f'Name {name} already exists.')
        pbar = _manager.counter(
                        desc=name,
                        total=total,
                        unit=unit,
                        leave=False)
        self._pbars[name] = pbar
    
    def __eq__(self, other):
        """Stages are equal if they have the same layout."""
        if not isinstance(other, _Stage):
            return False
        if (self._name != other._name) or (self._num_steps != other._num_steps):
            return False
        if len(self._stages) != len(other._stages):
            return False
        for s1, s2 in zip(self._stages, other._stages):
            if s1 != s2:
                return False
        return True
    
    @property
    def current_stage(self):
        return self.root._current_stage

class Tracker:

    def __init__(self):
        self.clear_best()
        self._stage = _Stage('_main')

    def add_stage(self, name, num_steps=1):
        return self._stage.add_stage(name, num_steps=num_steps)

    def clear_best(self):
        self.best_score = None
        self.best_stage = None
    
    def state_dict(self):
        ret = {'_stage': self._stage.state_dict(), 'best_score': self.best_score, 'best_stage': self.best_stage}
        return ret
    
    def load_state_dict(self, state_dict):
        self._stage.load_state_dict(state_dict['_stage'])
        self.best_score = state_dict['best_score']
        self.best_stage = state_dict['best_stage']

    @property
    def stage(self):
        return self._stage.current_stage
    
    def copy_(self, other):
        assert isinstance(other, Tracker)
        if self._stage != other._stage:
            logging.error('Somehow the stages are not identical. Abort copying stage.')
        else:
            self._stage.copy_(other._stage)
        self.best_score = other.best_score
        self.best_stage = other.best_stage

    def update_best(self, score, mode='min', quiet=False):
        """Update the best score and best stage. 
        
        Args:
            score: score for the current stage
            mode (str, optional): take the maximum or the minimum as the best score. Defaults to 'min'.
            quiet (bool, optional): flag to suppress outputting the best score. Defaults to False.
        
        Returns:
            updated (bool): whether the best score has been updated or not
        """
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
            self.best_stage = self.stage
        if self.best_score is not None and not quiet:
            logging.info(f'Best score is {self.best_score:.3f} at stage {self.best_stage}')
        return updated
    
    def update(self):
        self._stage.update()
