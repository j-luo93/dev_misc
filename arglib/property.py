import logging
from functools import wraps
from inspect import signature

import parser


def add_properties(*names):

    def decorator(cls):
        for name in names:
            assert not hasattr(cls, name), name
            setattr(cls, name, property(lambda self, name=name: getattr(self, f'_{name}'))) # NOTE The keyword is necessary.
        return cls

    return decorator

def set_properties(*names, **values):

    def decorator(self):
        for name in names:
            try:
                setattr(self, f'_{name}', values[name])
            except KeyError:
                logging.warning(f'{name} not passed by values')

        return self
        
    return decorator

def has_properties(*names):

    def decorator(cls):
        cls = add_properties(*names)(cls)
        old_init = cls.__init__

        @wraps(old_init)
        def new_init(self, *args, **kwargs):

            func_sig = signature(old_init)
            bound = func_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            all_args = bound.arguments
            self = set_properties(*names, **all_args)(self)
            old_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator

def use_arguments_as_properties(*names):
    def decorator(cls):
        cls = add_properties(*names)(cls)

        old_init = cls.__init__
        def new_init(self, *args, **kwargs):
            values = {name: parser.get_argument(name) for name in names}
            self = set_properties(*names, **values)(self)
            old_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator
