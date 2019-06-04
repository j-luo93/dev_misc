from dataclasses import dataclass

_REGS = dict()
def create_registry(name):
    assert name not in _REGS
    reg = Registry()
    _REGS[name] = reg
    return reg

class Registry(dict):

    def register(self, cls):
        cls_name = cls.__name__
        assert cls_name not in self
        cls = dataclass(cls)
        self[cls_name] = cls
        return cls