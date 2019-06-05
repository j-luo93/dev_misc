import logging
import sys
from pathlib import Path
from pprint import pformat

from pytrie import SortedStringTrie

from .argument import Argument, canonicalize, UnparsedArgument
from .property import has_properties, add_properties, set_properties


class DuplicateError(Exception):
    pass

class MultipleMatchError(Exception):
    pass

class KeywordError(Exception):
    pass

class ParsedError(Exception):
    pass

_NODES = dict()
@has_properties('command_name')
class _ParserNode:
    '''
    Each node is (sub)command.
    '''
    _unsafe = False
    def __init__(self, command_name):
        assert command_name not in _NODES
        _NODES[command_name] = self
        self._args = SortedStringTrie()
        self._arg_views = SortedStringTrie() # This stores argument views for booleans.
        self._registry = None
        self._kwds = {'--unsafe', '-u', '--help', '-h', '--config', '-cfg'}
        self._parsed = False
        self._cli_unparsed = None
        self.add_argument('--unsafe', '-u', dtype=bool, force=True)
    
    def __repr__(self):
        return self.command_name
    
    def add_cfg_registry(self, registry):
        self._registry = registry
        self.add_argument('--config', '-cfg', dtype=str, default='', force=True)
        
    def _check_keywords(self, name):
        # Raise error if keywords are used.
        if name in self._kwds:
            raise KeywordError(f'Keyword {name} cannot be used here')
        
    def add_argument(self, full_name, short_name=None, default=None, dtype=None, nargs=None, help='', force=False):
        if self._parsed and not _ParserNode._unsafe:
            raise ParsedError('Already parsed.')

        if not force:
            self._check_keywords(full_name)
            self._check_keywords(short_name)

        arg = Argument(full_name, short_name=short_name, default=default, dtype=dtype, nargs=nargs, help=help)
        if full_name in self._args:
            raise DuplicateError(f'Full name {full_name} has already been defined.')
        if short_name and short_name in self._args:
            raise DuplicateError(f'Short name {short_name} has already been defined.')
        
        self._args[full_name] = arg
        if short_name:
            self._args[short_name] = arg
        # For bool arguments, we need multiple views of the same underlying argument.
        if dtype is bool:
            pos_name = full_name
            neg_name = f'--no_{full_name[2:]}'
            self._arg_views[pos_name] = arg.view(pos_name, True)
            self._arg_views[neg_name] = arg.view(neg_name, False)
            if short_name:
                pos_name = short_name
                neg_name = f'-no_{short_name[1:]}'
                self._arg_views[pos_name] = arg.view(pos_name, True)
                self._arg_views[neg_name] = arg.view(neg_name, False)
        
        if self._parsed and _ParserNode._unsafe:
            self._parse_one_arg(full_name)
        return arg
    
    def help(self):
        print('Usage:')
        for k, a in self._args.items():
            print(' ' * 9, a)
        sys.exit(0)
        
    def get_argument(self, name, view_ok=True):
        name = canonicalize(name)
        # Always go for views first.
        a = None
        if view_ok:
            try:
                a = self._get_argument_from_trie(name, self._arg_views)
            except NameError:
                pass
        if a is None:
            a = self._get_argument_from_trie(name, self._args)
        return a
    
    def _get_argument_from_trie(self, name, trie):
        a = trie.values(prefix=name)
        if len(a) > 1:
            raise MultipleMatchError('Found multiple matches.')
        elif len(a) == 0:
            if _ParserNode._unsafe:
                return None
            else:
                raise NameError(f'Name {name} not found.')
        a = a[0]
        return a

    def _update_arg(self, arg, un_arg):
        if arg.is_view():
            arg.use_this_view()
        else:
            arg.value = un_arg.value

    def _parse_one_cli_arg(self, un_arg):
        """Parse one CLI argument.
        
        Args:
            un_arg (UnparsedArgument): unparsed argument.
        """
        a = self.get_argument(un_arg.name)
        if a is None:
            return None
        self._update_arg(a, un_arg)
        return a
    
    def _parse_one_arg(self, name):
        """Parse one declared argument.
        
        Args:
            name (str): the name of the declared argument. Fuzzy match is allowed.
        
        Returns:
            None or Argument: if an argument is updated, return that argument, otherwise return None.
        """
        arg = self.get_argument(name)
        if arg is None:
            return None

        prefixes = list()
        if arg.is_view():
            for view in arg.views:
                prefixes.append(view.name)
        else:
            prefixes.append(arg.full_name)
            if arg.short_name:
                prefixes.append(arg.short_name)
        items = list()
        for p in prefixes:
            items += list(self._cli_unparsed.iter_prefix_items(p))

        last_idx = None
        for k, un_arg in items:
            double_check = self.get_argument(un_arg.name)
            assert double_check == arg, f'{double_check} : {arg}'
            if last_idx is None or un_arg.idx > last_idx:
                last_idx = un_arg.idx
                self._update_arg(double_check, un_arg) # NOTE Use double_check here since this is the view that matches the un_arg.
            del self._cli_unparsed[k]
        if last_idx is None:
            return None
        return arg

    def _parse_cfg_arg(self, name, value):
        a = self.get_argument(name)
        a.value = value
        
    def parse_args(self):
        """
        There are three ways of parsing args. 
        1. Provide the declared argument (and its full name as the key) and find matching CLI arguments (unparsed). This is handled by ``_parse_one_arg`` function.
        2. Provide the CLI arguments, and find matching declared arguments. This is handled by ``_parse_one_cli_arg`` function.
        3. Read from config file. Handled by ``_parse_cfg_arg``.
        The second one is more natural, and we can easily go from left to right to make sure every CLI argument is handled.
        However, the first one is needed in unsafe mode, where a newly declared argument should be resolved.
        """
        if self._parsed:
            raise ParsedError('Already parsed.')

        argv = sys.argv[1:]
        self._cli_unparsed = SortedStringTrie()
        i = 0
        while i < len(argv):
            name = argv[i]
            value = tuple()
            j = i + 1
            while j < len(argv) and not argv[j].startswith('-'):
                value += (argv[j], )
                j += 1
            unparsed_a = UnparsedArgument(name, value)
            self._cli_unparsed[name] = unparsed_a
            i = j

        # Deal with help.
        if '--help' in argv or '-h' in argv:
            self.help()
        
        # Switch on unsafe mode (arguments can be created ad-hoc).
        if any([self._parse_one_arg('--unsafe'), self._parse_one_arg('-u')]):
            _ParserNode._unsafe = True
            logging.warning('Unsafe argument mode switched on.')

        # Use args in the cfg file as defaults.
        if self._registry is not None:
            a_cfg = self._parse_one_arg('--config')
            cfg_cls = self._registry[a_cfg.value]
            cfg = cfg_cls()
            default_args = vars(cfg)
            for name, v in default_args.items():
                self._parse_cfg_arg(name, v)

        # Use CLI args to override all.
        for un_arg in self._cli_unparsed.values():
            self._parse_one_cli_arg(un_arg)

        self._parsed = True
        return self._args

def _get_node(node):
    node = node or '_root'
    if len(_NODES) == 0:
        _NODES['_root'] = _ParserNode('_root')
    return _NODES[node]

def add_argument(full_name, short_name=None, default=None, dtype=None, node=None, nargs=None, help=''):
    node = _get_node(node)
    a = node.add_argument(full_name, short_name=short_name, default=default, dtype=dtype, nargs=nargs, help=help)
    return a.value

def get_argument(name, node=None):
    node = _get_node(node)
    return node.get_argument(name, view_ok=False).value # NOTE This public API should not allow views.

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

def use_arguments_as_properties(*names):
    def decorator(cls):
        cls = add_properties(*names)(cls)

        old_init = cls.__init__
        def new_init(self, *args, **kwargs):
            values = {name: get_argument(name) for name in names}
            self = set_properties(*names, **values)(self)
            old_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator