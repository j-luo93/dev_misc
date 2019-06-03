"""
Options or arguments are all referred to as arguments in this library. After all, options are just optinal arguments.
"""

class Argument:

    def __init__(self, full_name, short_name=None, default=None, type=str):
        """Construct an Argument object.
        
        Args:
            full_name (str): full name of this argument.
            short_name (str, optional): short name for this argument. Defaults to None.
        """
        self._full_name = full_name
        self._short_name = short_name
        self._value = default
        self._type = type
    
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = self._type(new_value)

    @property
    def short_name(self):
        return self._short_name
    
    @property
    def full_name(self):
        return self._full_name
