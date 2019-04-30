import os
import functools
from collections import OrderedDict, Callable, Hashable

import numpy as np
import torch

def get_tensor(data, dtype=None, requires_grad=False, use_cuda=True):
    use_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', False) and use_cuda # NOTE only use cuda when it's not overriden and there is a device available

    # If data is a tensor already, move to gpu if use_cuda
    if use_cuda and isinstance(data, torch.Tensor):
        return data.cuda()

    if dtype is None: # NOTE infer dtype
        dtype = 'f'
        if isinstance(data, np.ndarray) and issubclass(data.dtype.type, np.integer):
            dtype = 'l'

    # NOTE directly declare data. I believe it's faster on cuda, although I'm not entirely sure
    requires_grad = requires_grad and dtype == 'f'
    assert dtype in ['f', 'l']
    if use_cuda:
        module = getattr(torch, 'cuda')
    else:
        module = torch
    if dtype == 'f':
        cls = getattr(module, 'FloatTensor')
        dtype = 'float32'
    elif dtype == 'l':
        cls = getattr(module, 'LongTensor')
        dtype = 'int64'
    ret = cls(np.asarray(data, dtype=dtype))
    ret.requires_grad = requires_grad
    return ret

def get_zeros(*shape, **kwargs):
    if len(shape) == 1 and isinstance(shape[0], torch.Size): # NOTE deal with 1D tensor whose shape cannot be unpacked
        shape = list(shape[0])
    return get_tensor(np.zeros(shape), **kwargs)

def get_eye(n):
    return get_tensor(np.eye(n))

def counter_enumerate(iterable, *args, interval=1000, **kwargs):
    for i, item in enumerate(iterable, *args, **kwargs):
        yield i, item
        if i % interval == 0:
            print(f'\r{i}', end='')
            sys.stdout.flush()
    print('\nFinished enumeration')

def freeze(mod):
    for p in mod.parameters():
        p.requires_grad = False
    for m in mod.children():
        freeze(m)

'''
Modified from https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
First time cache it,
'''
class _Cache(object):

    CACHED_REPO = list()

    @classmethod
    def clear_all(cls):
        for func in _Cache.CACHED_REPO:
            func._cache = dict()

    def __init__(self, func, full=True):
        _Cache.CACHED_REPO.append(func)
        self.func = func
        self.full = full

        self.func._cache = dict() # NOTE insert a cache into func

    def __call__(self, *args, **kwargs):
        if self.full:
            '''
            Note that only args are used as keys for caching. kwargs are used for computation, but not for caching.
            '''
            if not isinstance(args, Hashable):
                # uncacheable. a list, for instance.
                # better to not cache than blow up.
                return self.func(*args, **kwargs)
            if args in self.func._cache:
                return self.func._cache[args]
            else:
                value = self.func(*args, **kwargs)
                self.func._cache[args] = value
                return value
        else:
            if self.func._cache:
                return self.func._cache[None]
            else:
                value = self.func(*args, **kwargs)
                self.func._cache[None] = value
                return value

    def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__

    def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def cache(full=True):
    return lambda func: _Cache(func, full=full)

def clear_cache():
    _Cache.clear_all()

def norm(x):
    return torch.nn.functional.normalize(x, dim=-1)
