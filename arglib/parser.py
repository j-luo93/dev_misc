import logging
import sys
from pathlib import Path
from pprint import pformat

from .argument import Argument, get_format, SmartType

class DuplicateError(Exception):
    pass

class MultipleMatchError(Exception):
    pass

class KeywordError(Exception):
    pass

_NODES = dict()
class _ParserNode:
    '''
    Each node is (sub)command.
    '''
    _unsafe = False
    def __init__(self, command_name):
        assert command_name not in _NODES
        _NODES[command_name] = self
        self._command_name = command_name
        self._args = dict()
        self._short_args = dict() # NOTE Keep all short names as well.
        self._registry = None
        self._kwds = {'--unsafe', '-u', '--help', '-h', '--config', '-cfg'}
    
    def __repr__(self):
        return self.command_name
    
    def add_cfg_registry(self, registry):
        self._registry = registry
        self.add_argument('--config', '-cfg', dtype=str, default='', force=True)
        
    @property
    def command_name(self):
        return self._command_name
    
    def _check_keywords(self, name):
        # Raise error if keywords are used.
        if name in self._kwds:
            raise KeywordError(f'Keyword {name} cannot be used here')
        
    def add_argument(self, full_name, short_name=None, default=None, dtype=None, help='', force=False):
        if not force:
            self._check_keywords(full_name)
            self._check_keywords(short_name)

        arg = Argument(full_name, short_name=short_name, default=default, dtype=dtype, help=help)
        if full_name in self._args:
            raise DuplicateError(f'Full name {full_name} has already been defined.')
        if short_name and short_name in self._short_args:
            raise DuplicateError(f'Short name {short_name} has already been defined.')
        self._args[full_name] = arg
        if short_name:
            self._short_args[short_name] = arg
        return arg
    
    def get_argument(self, name, default=None):
        fmt = get_format(name)
        if fmt == 'plain': # Default to full name.
            fmt = 'full'
            name = f'--{name}'
        if fmt in ['full', 'short']:
            return self._get_argument(name, default=default, by=fmt)
    
    def _get_argument(self, name, default=None, by='full'):
        assert by in ['full', 'short']
        dict_ = self._args if by == 'full' else self._short_args
        # Use exact match first.
        try:
            return dict_[name]
        except KeyError:
            ret = None
            attr_name = f'{by}_name'
            for arg in dict_.values():
                if getattr(arg, attr_name).startswith(name):
                    if ret is not None:
                        raise MultipleMatchError('Found multiple matches.')
                    ret = arg
            if ret is None:
                if _ParserNode._unsafe:
                    logging.warning(f'Name {name} not found. Adding {name} in unsafe mode.')
                    ret = self.add_argument(name, force=True, dtype=SmartType, default=default)
                else:
                    raise NameError(f'Name {name} not found.')
            return ret

    # def _safe_update_argument(self, arg):
    #     full_name = arg.full_name
    #     if not full_name in self._args:
    #         logging.warning(f'Argument named {full_name} does not exist. Adding it now.')
    #         self.add_argument(full_name, default=arg.value, dtype=arg.dtype)
    
    def _update_argument(self, full_name, value):
        if full_name not in self._args:
            raise NameError(f'Full name {full_name} not found.')
        else:
            self._args[full_name].value = value
 
    def help(self):
        print('Usage:')
        for k, a in sorted(self._args.items()):
            print(' ' * 9, a)
        sys.exit(0)
        
    def parse_args(self):
        argv = sys.argv[1:]
        # Deal with help.
        if '--help' in argv or '-h' in argv:
            self.help()
        
        # Switch on unsafe mode (arguments can be created ad-hoc).
        if '--unsafe' in argv or '-u' in argv:
            _ParserNode._unsafe = True
            argv = list(filter(lambda x: x not in ['--unsafe', '-u'], argv))
            logging.warning('Unsafe argument mode on.')

        # Go through all args once first. Always take the last value if multiple values are specified for the same argument.
        i = 0
        cli_parsed = dict()
        while i < len(argv):
            name = argv[i]
            a = self.get_argument(name)
            i += 1
            v = argv[i]
            cli_parsed[a.full_name] = v
            i += 1
        
        # Use args in the cfg file as defaults.
        if self._registry is not None:
            a_cfg = self.get_argument('--config')
            cfg_cls = self._registry[a_cfg.value]
            cfg = cfg_cls()
            default_args = vars(cfg)
            for k, v in default_args.items():
                full_name = f'--{k}'
                self._update_argument(full_name, v)

        # Use CLI args to override all.
        for full_name, v in cli_parsed.items():
            a = self.get_argument(full_name)
            a.value = v

        return self._args

def _get_node(node):
    node = node or '_root'
    if len(_NODES) == 0:
        _NODES['_root'] = _ParserNode('_root')
    return _NODES[node]

def add_argument(full_name, short_name=None, default=None, dtype=None, node=None, help=''):
    node = _get_node(node)
    a = node.add_argument(full_name, short_name=short_name, default=default, dtype=dtype, help=help)
    return a.value

def get_argument(name, node=None, default=None):
    node = _get_node(node)
    return node.get_argument(name, default=default).value

def clear_parser():
    global _NODES
    _NODES = dict()
    _ParserNode._unsafe = False

def parse_args(node=None):
    node = _get_node(node)
    args = node.parse_args()
    return {a.name: a.value for k, a in sorted(args.items())}

def add_cfg_registry(registry, node=None):
    node = _get_node(node)
    node.add_cfg_registry(registry)
