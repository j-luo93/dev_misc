from functools import wraps
from typing import List

import torch

# NOTE(j_luo) Magic methods cannot be detected through __getattr__, so they have to be provided here.
_action2attrs = {
    'inherit': {'log_softmax', 'softmax', '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__'}
}
_attr2action = dict()
for action, attrs in _action2attrs.items():
    for attr in attrs:
        _attr2action[attr] = action


class NamedTensor:

    def __init__(self, tensor, *, names: List[str] = None):
        if names is None:
            raise TypeError(f'Must explicitly pass names.')
        if not torch.is_tensor(tensor):
            raise TypeError(f'Expect to create a NamedTensor from a Tensor object, but got {type(tensor)}.')
        if tensor.dim() != len(names):
            raise ValueError(f'Expect names to have the same length as the number of dimensions of tensor.')
        self._tensor = tensor
        self.names = tuple(names)

    @property
    def tensor(self):
        return self._tensor

    def __getattr__(self, attr):
        ret = getattr(self._tensor, attr)
        if callable(ret) and attr not in _attr2action:
            logging.warning(f'{attr} is a callable, but not taken care of.')
        return ret

    def __repr__(self):
        names = ', '.join([f'"{name}"' for name in self.names])
        out = f'NamedTensor({names}):\n'
        out += repr(self._tensor)
        return out


def get_wrapper(attr, *, action=None):
    assert action is not None
    assert action in ['inherit']
    func = getattr(torch.Tensor, attr)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        ret = func(self.tensor, *args, **kwargs)
        if action == 'inherit':
            if not torch.is_tensor(ret):
                raise TypeError(f'Expecting a tensor instance, but got {type(ret)}.')
            ret = NamedTensor(ret, names=self.names)
        return ret

    return wrapper


for attr, action in _attr2action.items():
    if hasattr(NamedTensor, attr):
        raise ValueError(f'An attribute named {attr} already exists.')
    setattr(NamedTensor, attr, get_wrapper(attr, action=action))
