import logging
from functools import wraps
from inspect import signature


def has_property(*names):

    def decorator(cls):
        for name in names:
            assert not hasattr(cls, name), name
            setattr(cls, name, property(lambda self, name=name: getattr(self, f'_{name}'))) # NOTE The keyword is necessary.

        old_init = cls.__init__

        @wraps(old_init)
        def new_init(self, *args, **kwargs):

            func_sig = signature(old_init)
            bound = func_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            all_args = bound.arguments
            for name in names:
                try:
                    setattr(self, f'_{name}', all_args[name])
                except KeyError:
                    logging.warning(f'{name} not passed to __init__ call.')

            old_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator
