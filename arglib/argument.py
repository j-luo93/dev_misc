"""
Options or arguments are all referred to as arguments in this library. After all, options are just optinal arguments.
"""

class FormatError(Exception):
    pass

def _check_format(name):
    if name.startswith('--') and not name.startswith('---'):
        return 'full'
    if name.startswith('-') and not name.startswith('--'):
        return 'short'
    if not name.startswith('-'):
        return 'plain'
    raise FormatError(f'Unrecognized format for name {name}')

def canonicalize(name):
    fmt = _check_format(name)
    if fmt == 'plain':
        return f'--{name}'
    else:
        return name

class UnparsedArgument:
    
    _idx = 0
    def __init__(self, name, value):
        if _check_format(name) == 'plain':
            raise Exception('Should not have come here. Something wrong with the parser.')
        self._name = name
        assert isinstance(value, tuple)
        if len(value) == 0:
            value = None
        elif len(value) == 1:
            value = value[0]
        self._value = value
        self._idx = UnparsedArgument._idx
        UnparsedArgument._idx += 1
    
    @property
    def idx(self):
        return self._idx

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self._name

class Argument:

    def __init__(self, full_name, short_name=None, default=None, dtype=None, help=''):
        """Construct an Argument object.
        
        Args:
            full_name (str): full name of this argument.
            short_name (str, optional): short name for this argument. Defaults to None.
        """
        if _check_format(full_name) != 'full':
            raise FormatError(f'Format wrong for full name {full_name}.')
        if short_name is not None and _check_format(short_name) != 'short':
                raise FormatError(f'Format wrong for short name {short_name}.')

        self._full_name = full_name
        self._short_name = short_name
        self._help = help
        self._dtype = dtype
        self._default = default

        self.value = default
        if self.default is not None and dtype is None:
            # Use the type of the default.
            self._dtype = type(self.default)
    
    @property
    def default(self):
        return self._default

    @property
    def help(self):
        return self._help
    
    @property
    def dtype(self):
        return self._dtype    

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if self._dtype:
            new_value = self._dtype(new_value)
        self._value = new_value

    @property
    def short_name(self):
        return self._short_name
    
    @property
    def full_name(self):
        return self._full_name

    @property
    def name(self):
        """This is the plain name without leading hyphen(s)."""
        return self.full_name[2:]

    def __str__(self):
        out = f'{self.full_name}'
        if self.short_name:
            out += f' {self.short_name}'
        if self.dtype:
            out += f' ({self.dtype.__name__})'
        if self.help:
            out += f': {self.help}'
        if self.default is not None:
            out += f' [DEFAULT = {self.default}]'
        return out
        
    def __repr__(self):
        return f'Argument({self.name})'