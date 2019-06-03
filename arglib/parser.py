import sys
from pprint import pformat

from .argument import Argument


class DuplicateError(Exception):
    pass

class MultipleMatchError(Exception):
    pass

_NODES = dict()
class _ParserNode:
    '''
    Each node is (sub)command.
    '''
        
    def __init__(self, command_name):
        assert command_name not in _NODES
        _NODES[command_name] = self
        self._command_name = command_name
        self._args = dict()
        self._short_names = set() # NOTE Keep all short names.
    
    def __repr__(self):
        return self.command_name
        
    @property
    def command_name(self):
        return self._command_name
    
    def __str__(self):
        args = {k: a.value for k, a in self._args.items()}
        return pformat(args)

    def add_argument(self, full_name, short_name=None, default=None, type=str):
        if not full_name.startswith('--') or full_name.startswith('---'):
            raise ValueError(f'Format wrong for full name {full_name}')
        if short_name:
            if not short_name.startswith('-') or short_name.startswith('--'):
                raise ValueError(f'Format wrong for short name {short_name}')
        arg = Argument(full_name, short_name=short_name, default=default, type=type)
        if full_name in self._args:
            raise DuplicateError(f'Full name {full_name} has already been defined.')
        if short_name and short_name in self._short_names:
            raise DuplicateError(f'Short name {short_name} has already been defined.')
        self._args[full_name] = arg
        if short_name:
            self._short_names.add(short_name)
        return arg
    
    def get_argument(self, name):
        if name.startswith('--'): # Search by full name.
            return self._get_argument(name, by='full')
        elif name.startswith('-'):
            return self._get_argument(name, by='short')
        else:
            raise NameError('Unrecognized argument name')
    
    def _get_argument(self, name, by='full'):
        assert by in ['full', 'short']
        ret = None
        for arg in self._args.values():
            if by == 'full' and arg.full_name.startswith(name):
                if ret is not None:
                    raise MultipleMatchError('Found multiple matches')
                ret = arg
            elif by == 'short' and arg.short_name and arg.short_name.startswith(name):
                if ret is not None:
                    raise MultipleMatchError('Found multiple matches')
                ret = arg
        return ret

def _get_node(node):
    node = node or '_root'
    if len(_NODES) == 0:
        _NODES['_root'] = _ParserNode('_root')
    return _NODES[node]

def add_argument(full_name, short_name=None, default=None, type=str, node=None):
    node = _get_node(node)
    return node.add_argument(full_name, short_name=short_name, default=default, type=type)

def get_argument(name, node=None):
    node = _get_node(node)
    return node.get_argument(name)        

def clear_parser():
    global _NODES
    _NODES = dict()

def parse_args():
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        name = argv[i]
        a = get_argument(name)
        i += 1
        v = argv[i]
        a.value = v
        i += 1
    return _get_node('_root')
