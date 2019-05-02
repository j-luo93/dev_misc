import functools
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable

from hrr.utils.map import Map

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

class StructuredCache:
    
    def __init__(self):
        # What should be kept.
        self._to_keep = dict()
        # What should be cached.
        self._to_cache = defaultdict(set)
        # What is cached. Note that each instance method has its own cache, keyed by the object's unique id. 
        self.clear_cache()
        # The mapping from the object id to the object.
        self._id2obj = dict()
    
    def keep(self, name, ret):
        '''Only keep what should be kept.'''
        if self._to_keep[name] is None:
            return ret
        else:
            return Map(**{k: ret[k] for k in self._to_keep[name] if k in ret})
    
    def __contains__(self, key):
        return key in self._to_keep
    
    def register_keep(self, name, *to_keep):
        '''Record what to keep for each function.'''
        assert name not in self
        if len(to_keep) == 0:
            self._to_keep[name] = None # NOTE None means keeping everthing. 
        else:
            self._to_keep[name] = set(to_keep)
    
    def register_cache(self, name, *keys):
        '''Register cache for all instances of the same registered function with ``name``.'''
        assert name in self._to_keep
        self._to_cache[name].update(keys)
    
    def cache(self, name, obj, ret):
        id_ = id(obj)
        if id_ not in self._id2obj:
            self._id2obj[id_] = obj
        else:
            assert self._id2obj[id_] is obj # Make sure it's the same object.
        for k in self._to_cache[name]:
            assert k not in self._cache[name][id_]
            self._cache[name][id_][k] = ret[k]
    
    def get_cache(self, name, *keys):
        '''Get all caches generated from the same function.'''
        ret = list()
        for id_ in self._cache[name]:
            obj = self._id2obj[id_]
            cache = self._cache[name][id_]
            ret.append((obj, {k: cache[k] for k in keys}))
        return ret
    
    def clear_cache(self):
        self._cache = defaultdict(lambda: defaultdict(defaultdict))

_SC = StructuredCache()
def sc(name, *to_keep):
    global _SC
    
    def descriptor(func):
        def decorator(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            assert isinstance(ret, Map)
            _SC.cache(name, self, ret)
            return _SC.keep(name, ret)

        return decorator
    
    _SC.register_keep(name, *to_keep)
    return descriptor

def sc_clear_cache():
    global _SC
    _SC.clear_cache()

def sc_register_cache(name, *keys):
    global _SC
    _SC.register_cache(name, *keys)
    
def sc_get_cache(name, *keys):
    global _SC
    return _SC.get_cache(name, *keys)
