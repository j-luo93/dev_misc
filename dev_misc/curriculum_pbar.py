import logging
import time
from functools import wraps

from enlighten import Counter

from arglib import has_properties

from .logger import log_this


class PBarFinishedError(Exception):
    pass


def get_c_prop(name):
    return getattr(CurriculumPBar, name)


def run_cond_c_prop(name, cond, default=None):
    assert isinstance(cond, bool)

    def decorator(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            c_prop = _get_bool_c_prop(name)
            if c_prop == cond:
                return func(*args, **kwargs)
            else:
                if default is None:
                    return None
                return default()

        return wrapped

    return decorator


def run_if_c_prop(name, default=None):
    return run_cond_c_prop(name, True, default=default)


def run_unless_c_prop(name, default=None):
    return run_cond_c_prop(name, False, default=default)


def _get_bool_c_prop(name):
    c_prop = get_c_prop(name)
    assert isinstance(c_prop, bool)
    return c_prop


def context_if_c_prop(name, context):

    def decorator(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            c_prop = _get_bool_c_prop(name)
            if c_prop:
                with context():
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapped

    return decorator


@has_properties('name')
class CurriculumPBar(Counter):
    """This is an enhanced version of the `Counter` class in `enlighten`.

    It has two additional features:
    1. call callback functions when pbar is full.
    2. some registered curriculum properties `CurriculumProperty` can be tracked globally.
    """

    def __init__(self, name=None, once=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_callbacks = list()
        self._post_callbacks = list()
        self._once = once
        if name is None:
            raise NameError(f'Pass name explictly.')
        if name in _C_PBAR_NAMES:
            raise NameError(f'Name {name} already exists.')
        _C_PBAR_NAMES[name] = self
        self._prop_names = list()

    def state_dict(self):
        ret = {'count': self.count}
        prop_ret = dict()
        for prop_name in self._prop_names:
            prop_ret[prop_name] = getattr(self, prop_name)
        ret['props'] = prop_ret
        return ret

    def load_state_dict(self, state_dict):
        self.count = state_dict['count']
        for prop_name in self._prop_names:
            value = state_dict['props'][prop_name]
            setattr(self, prop_name, value)

    # TODO I don't like `once`.
    @property
    def once(self):
        return self._once

    def add_callback(self, callback, when):
        if when == 'initial':  # TODO This shouldn't be considered as a callback, right?
            callback()
            return
        callbacks = self._pre_callbacks if when == 'before' else self._post_callbacks
        callbacks.append(callback)

    def add_inc_callback(self, name, when, inc=1):
        if not hasattr(self, name):
            self.add_property(name)

        def inc_func():
            value = getattr(self, name)
            setattr(self, name, value + inc)
        self.add_callback(inc_func, when)

    def add_anneal_callback(self, name, decay, when, min_value=None):
        @log_this('IMP', msg='Annealing')
        def anneal():
            value = getattr(self, name)
            new_value = value * decay
            if min_value is not None:
                new_value = max(min_value, new_value)
            setattr(self, name, new_value)
        self.add_callback(anneal, when)

    def add_set_value_callback(self, name, value, when):
        def set_value():
            setattr(self, name, value)
        self.add_callback(set_value, when)

    def add_property(self, prop_name):
        if hasattr(type(self), prop_name):
            raise NameError(f'Name "{prop_name}" has already been used.')
        prop = _C_PROP_NAMES[prop_name]
        setattr(type(self), prop_name, prop)
        self._prop_names.append(prop_name)

    def add_switch(self, name, before_value, after_value):
        """Add a pbar and two callbacks (one before and one after) to simulate a switch. This esssentially wraps one `add_property` call and two `add_set_value_callback` calls.

        Args:
            name (str): the name of the `CurriculumProperty` object that controls the switch
            before_value (any): the value to be set for callback (before)
            after_value (any): the value to be set for callback (after)
        """
        self.add_property(name)
        self.add_set_value_callback(name, before_value, 'before')
        self.add_set_value_callback(name, after_value, 'after')

    def reset(self):
        self.count = 0
        self.start = time.time()
        for callback in self._pre_callbacks:
            callback()

    def update(self):
        if self.count == self.total:
            if self.once:
                raise PBarFinishedError
            self.reset()
        super().update()
        if self.count == self.total:
            for callback in self._post_callbacks:
                callback()

    @property
    def finished(self):
        return self.once and self.count == self.total


_C_PROP_NAMES = dict()


def clear_c_props():
    global _C_PROP_NAMES
    for name in _C_PROP_NAMES:
        if hasattr(CurriculumPBar, name):
            delattr(CurriculumPBar, name)
    _C_PROP_NAMES = dict()


_C_PBAR_NAMES = dict()


def clear_c_pbars():
    global _C_PBAR_NAMES
    _C_PBAR_NAMES = dict()


def get_c_pbar(name):
    return _C_PBAR_NAMES[name]


class CurriculumProperty:
    """Any declared instance of this class would be owned by a pbar and the original class for which it is originally declared.
    However, only the pbar can both set and get the property whereas the original class can only get the property.
    """

    def __init__(self, name):
        if name in _C_PROP_NAMES:
            raise NameError(f'Name "{name}" has already been used for a curriculum property.')
        self.name = name
        _C_PROP_NAMES[name] = self

    def __get__(self, instance, owner):
        return self._value

    # IDEA use `eval` to evaluate expressions.
    @log_this('IMP', msg='Setting curriculum property', arg_list=['self.name', 'value'])
    def __set__(self, instance, value):
        if not isinstance(instance, CurriculumPBar):
            raise TypeError(f'You cannot set a new value to this property unless you are a `CurriculumPBar` instance.')
        self._value = value
