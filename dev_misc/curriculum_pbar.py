import time
from functools import wraps

from enlighten import Counter


class PBarFinishedError(Exception):
    pass


def get_c_prop(name):
    try:
        return getattr(CurriculumPBar, name)
    except AttributeError:
        return None


def run_cond_c_prop(name, cond, default=None):
    assert isinstance(cond, bool)

    def decorator(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            c_prop = _get_bool_c_prop(name)
            if c_prop == cond:
                return func(*args, **kwargs)
            else:
                return default

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


class CurriculumPBar(Counter):
    """This is an enhanced version of the `Counter` class in `enlighten`.

    It has two additional features:
    1. call callback functions when pbar is full.
    2. some registered curriculum properties `CurriculumProperty` can be tracked globally.
    """

    def __init__(self, once=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = list()
        self._once = once

    @property
    def once(self):
        return self._once

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def add_inc_one_callback(self, name):
        def inc_one():
            value = getattr(self, name)
            setattr(self, name, value + 1)
        self.add_callback(inc_one)

    def add_set_value_callback(self, name, value):
        def set_value():
            setattr(self, name, value)
        self.add_callback(set_value)

    def add_property(self, prop_name):
        if hasattr(type(self), prop_name):
            raise NameError(f'Name "{prop_name}" has already been used.')
        prop = _C_PROP_NAMES[prop_name]
        setattr(type(self), prop_name, prop)

    def reset(self):
        self.count = 0
        self.start = time.time()

    def update(self):
        if self.count == self.total:
            if self.once:
                raise PBarFinishedError
            self.reset()
        super().update()
        if self.count == self.total:
            for callback in self._callbacks:
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

    def __set__(self, instance, value):
        if not isinstance(instance, CurriculumPBar):
            raise TypeError(f'You cannot set a new value to this property unless you are a `CurriculumPBar` instance.')
        self._value = value
