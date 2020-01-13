# TODO(j_luo) add tests


from typing import ClassVar, Dict

import torch

from dev_misc.utils import Singleton


class NamedDimension:

    def __init__(self, name: str, size: int):
        self._name = name
        self._size = size

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size


class SizeConflict(Exception):
    """Raise this error if the registered named dimension has a different size than currently needed."""


class NamedDimensionRegistry(Singleton):

    _names: ClassVar[Dict[str, NamedDimension]] = dict()

    def register_tensor(self, tensor: torch.Tensor):
        for name, size in zip(tensor.names, tensor.shape):
            if name is not None:
                self.register(name, size)

    def register(self, name: str, size: int) -> NamedDimension:
        cls = type(self)
        if name in cls._names:
            nd = cls._names[name]
            if nd.size != size:
                raise SizeConflict(
                    f'Asking for size {size} but a named dimension with size {nd.size} already registered.')
        else:
            nd = NamedDimension(name, size)
            cls._names[name] = nd
        return nd
