import logging
import time

import enlighten

from arglib import has_properties

from .metrics import plain

_manager = enlighten.get_manager()
_stage_names = set()
@has_properties('name', 'num_steps', 'parent')
class _Stage:

    def __init__(self, name, num_steps=1, parent=None):
        self._pbars = dict()
        self.substages = list()
        if self.num_steps > 1:
            self.add_pbar(name, total=self.num_steps)
    
    def update_pbars(self):
        for pbar in self._pbars.values():
            if pbar.total == pbar.count:
                pbar.count = 0
                pbar.start = time.time()
            pbar.update()

    def add_pbar(self, name, total=None, unit='samples'):
        if name in self._pbars:
            raise NameError(f'Name {name} already exists.')
        pbar = _manager.counter(
                        desc=name,
                        total=total,
                        unit=unit,
                        leave=False)
        self._pbars[name] = pbar
    
    def add_stage(self, name, num_steps=1):
        assert name not in _stage_names

        stage = _Stage(name, num_steps=num_steps, parent=self)
        _stage_names.add(name)
        self.substages.append(stage)
        return stage
    
    def __str__(self):
        return f'"{self.name}"'
    
    def __repr__(self):
        return f'Stage(name={self.name}, num_steps={self.num_steps})'
    
    def load_state_dict(self, state_dict):
        for name, pbar_meta in state_dict['_pbars'].items():
            pbar = self._pbars[name]
            pbar.count = pbar_meta['count']
            pbar.refresh()
        for s1, s2 in zip(self.substages, state_dict['_stages']):
            s1.load_state_dict(s2)
    
    def state_dict(self):
        ret = dict()
        ret['_pbars'] = {name: {'count': pbar.count} for name, pbar in self._pbars.items()} # NOTE pbar itself cannot be serialized for some reason.
        stage_ret = list()
        for s in self.substages:
            stage_ret.append(s.state_dict())
        ret['_stages'] = stage_ret
        return ret

@has_properties('step', 'substage_idx')
class _Node:
    
    def __init__(self, stage, step, substage_idx):
        self.stage = stage
    
    def is_last(self):
        last_step = (self.step == self.stage.num_steps - 1)
        if self.stage.substages:
            last_substage = (self.substage_idx == len(self.stage.substages) - 1)
            last = last_substage and last_step
        else:
            last = last_step
        return last
    
    def next_node(self):
        """Return whether next node will increment the step."""
        if self.stage.substages:
            new_substage_idx = self.substage_idx + 1
            incremented = False
            if new_substage_idx == len(self.stage.substages):
                new_substage_idx = 0
                incremented = True
            new_step = self.step + incremented
            return _Node(self.stage, new_step, new_substage_idx), incremented
        else:
            return _Node(self.stage, self.step + 1, None), True
    
    def __str__(self):
        return f'{self.stage}: {self.step}'
    
    def __repr__(self):
        return f'Node(stage={str(self.stage)}, step={self.step}, substage_idx={self.substage_idx})'
    
class _Path:

    def __init__(self, schedule):
        self._nodes = list()
        self._schedule = schedule
        self._get_first_path(self._schedule)

    def _add(self, node):
        # Check that this is a valid extension of the original path.
        if len(self._nodes) == 0:
            safe = True
        else:
            last_node = self._nodes[-1]
            safe = last_node.stage.substages[last_node.substage_idx] is node.stage
        assert safe
        # Add it.
        self._nodes.append(node)
    
    def __str__(self):
        ret = ' -> '.join([str(node) for node in self._nodes])
        return ret

    def _get_first_path(self, stage_or_node):

        def helper(stage_or_node):
            if isinstance(stage_or_node, _Stage):
                stage = stage_or_node
                if stage.substages:
                    self._add(_Node(stage, 0, 0))
                    helper(stage.substages[0])
                else:
                    self._add(_Node(stage, 0, None)) # None means there is no substage.
            else:
                assert isinstance(stage_or_node, _Node)
                node = stage_or_node
                if node.stage.substages:
                    new_node = _Node(node.stage, node.step, node.substage_idx)
                    self._add(new_node)
                    child_node = _Node(new_node.stage.substages[new_node.substage_idx], 0, 0)
                    helper(child_node)
                else:
                    self._add(node)
        
        helper(stage_or_node)

    def next_path(self):
        """Note that this is in-place. It returns the nodes incremented."""
        # First backtrack to the first ancestor that hasn't been completed yet.
        i = len(self._nodes)
        while i > 0:
            i -= 1
            last_node = self._nodes[i]
            if not last_node.is_last():
                break
        # Now complete it.
        if last_node.is_last():
            raise StopIteration('Cannot find next path.')
        else:
            affected_nodes = self._nodes[i + 1:] # NOTE Everything that is last will be incremented.
            self._nodes = self._nodes[:i]
            next_node, incremented = last_node.next_node()
            if incremented:
                affected_nodes.append(next_node)
            self._get_first_path(next_node)
        return affected_nodes

class _Schedule(_Stage):

    def __init__(self):
        super().__init__('_main', num_steps=1)
        self._path = None
    
    def update(self):
        if self._path is None:
            self._path = _Path(self)
        affected_nodes = self._path.next_path()
        for node in affected_nodes:
            node.stage.update_pbars()

class Tracker:

    def __init__(self):
        self.clear_best()
        self._schedule = _Schedule()

    def add_stage(self, name, num_steps=1):
        return self._schedule.add_stage(name, num_steps=num_steps)

    def clear_best(self):
        self.best_score = None
        self.best_stage = None
    
    def state_dict(self):
        ret = {'_schedule': self._schedule.state_dict(), 'best_score': self.best_score, 'best_stage': self.best_stage}
        return ret
    
    def load_state_dict(self, state_dict):
        self._schedule.load_state_dict(state_dict['_schedule'])
        self.best_score = state_dict['best_score']
        self.best_stage = state_dict['best_stage']
    
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
            self.best_stage = str(self.current_stage)
        if self.best_score is not None and not quiet:
            logging.info(f'Best score is {self.best_score:.3f} at stage {self.best_stage}')
        return updated
    
    def update(self):
        self._schedule.update()
    
    @property
    def current_stage(self):
        pass
        # TODO 
