import logging
import sys
from pathlib import Path
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
        self._registry = None
    
    def __repr__(self):
        return self.command_name
    
    def add_cfg_registry(self, registry):
        self._registry = registry
        self.add_argument('--config', '-cfg')
        
    @property
    def command_name(self):
        return self._command_name
    
    def add_argument(self, full_name, short_name=None, default=None, type=str):
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
            raise NameError('Unrecognized argument name.')
    
    def _get_argument(self, name, by='full'):
        assert by in ['full', 'short']
        ret = None
        for arg in self._args.values():
            if by == 'full' and arg.full_name.startswith(name):
                if ret is not None:
                    raise MultipleMatchError('Found multiple matches.')
                ret = arg
            elif by == 'short' and arg.short_name and arg.short_name.startswith(name):
                if ret is not None:
                    raise MultipleMatchError('Found multiple matches.')
                ret = arg
        if ret is None:
            raise NameError(f'Name {name} not found.')
        return ret

    def _check_exist(self, name):
        full_name = f'--{name}'
    
    def _safe_update_argument(self, arg):
        full_name = arg.full_name
        if not full_name in self._args:
            logging.warning(f'Argument named {full_name} does not exist. Adding it now.')
            self.add_argument(full_name, default=arg.value, type=arg.type)

    def parse_args(self):
        if self._registry is not None:
            a_cfg = self._args['--config']
            cfg_cls = self._registry[a_cfg]
            cfg = cfg_cls()
            default_args = vars(cfg)
            for k, v in default_args.items():
                a = Argument(f'--{k}', default=v, type=type(v))
                self._safe_update_argument(a)

        argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            name = argv[i]
            a = self.get_argument(name)
            i += 1
            v = argv[i]
            a.value = v
            i += 1
        
        return {a.name: a.value for k, a in self._args.items()}

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

def parse_args(node=None):
    node = _get_node(node)
    return node.parse_args()

def add_cfg_registry(registry, node=None):
    node = _get_node(node)
    node.add_cfg_registry(registry)
