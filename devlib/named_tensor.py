import ctypes
import inspect
import warnings
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from deprecated import deprecated

from devlib import get_range

Module = nn.Module
Tensor = torch.Tensor


@dataclass
class Patch:
    module: ModuleType
    unpatched: Callable
    patched: Callable


def _get_caller_name(stacklevel=1):
    frame = inspect.currentframe()
    while stacklevel > 0:
        frame = frame.f_back
        stacklevel -= 1
    return frame.f_code.co_name


class _Patcher:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            cls._instance = obj
        return cls._instance

    def __init__(self):
        self._patches: Dict[str, Patch] = dict()
        self._patched = False

    def patch(self, module: ModuleType):
        """Note that this doesn't actually patch the functions -- it just hooks them. To actually patch them, use `patch_named_tensors`."""

        def decorator(patched):
            name = patched.__name__
            unpatched = getattr(module, name)
            patch = Patch(module, unpatched, patched)
            if name in self._patches:
                raise NameError(f'Duplicate name "{name}".')
            self._patches[name] = patch
            return patched

        return decorator

    def call_unpatched(self, *args, stacklevel: int = 1, **kwargs):
        caller_name = _get_caller_name(stacklevel=stacklevel + 1)
        patch = self._patches[caller_name]
        return patch.unpatched(*args, **kwargs)

    def patch_named_tensors(self):
        if self._patched:
            raise RuntimeError(f'Already patched.')
        for name, patch in self._patches.items():
            setattr(patch.module, name, patch.patched)
        self._patched = True

    def unpatch_named_tensors(self):
        if not self._patched:
            raise RuntimeError(f'Not patched yet.')
        for name, patch in self._patches.items():
            setattr(patch.module, name, patch.unpatched)
        self._patched = False


_patcher = _Patcher()
patch = _patcher.patch
patch_named_tensors = _patcher.patch_named_tensors
unpatch_named_tensors = _patcher.unpatch_named_tensors
call_unpatched = _patcher.call_unpatched


@patch(torch.Tensor)
def refine_names(self, *args, **kwargs):
    ret = call_unpatched(self, *args, **kwargs)
    _incref = ctypes.pythonapi.Py_IncRef
    _incref.argtypes = [ctypes.py_object]
    for _ in range(len(args)):
        _incref(None)
    return ret


@patch(torch.nn.functional)
def leaky_relu(tensor: Tensor, *args, **kwargs):
    names = tensor.names
    ret = call_unpatched(tensor.rename(None), *args, **kwargs)
    return ret.refine_names(*names)


# -------------------------------------------------------------- #
#                      Old helper functions                      #
# -------------------------------------------------------------- #

# TODO(j_luo) Document name changes for the following helper functions.


class MustBeNamed(Exception):
    pass


def _check_names(tensor: Tensor) -> bool:
    """Check whether tensor is named."""
    if all([name is None for name in tensor.names]):
        raise MustBeNamed('Tensor must be named.')
    return tensor.names


@deprecated(reason='Old helper functions.')
def embed(mod: Module, tensor: Tensor, new_dim_name: str) -> Tensor:
    """Embed a tensor and adjust the names."""
    names = _check_names(tensor)
    new_names = names + (new_dim_name, )
    return mod(tensor.rename(None)).refine_names(*new_names)


@deprecated(reason='Old helper functions.')
def self_attend(mod: Module, tensor: Tensor, new_name: str) -> Tuple[Tensor, Tensor]:
    old_names = _check_names(tensor)
    new_names = old_names[:-1] + (new_name,)

    name_len, name_batch = old_names[:2]
    name_len_T = f'{name_len}_T'
    tensor = tensor.rename(None)
    output, weight = mod(tensor, tensor, tensor)
    output = output.refine_names(*new_names)
    weight = weight.refine_names(name_batch, name_len, name_len_T)
    return output, weight


@deprecated(reason='Old helper functions.')
def adv_index(tensor: Tensor, name: str, index: Tensor) -> Tensor:
    # TODO(j_luo) Expand this function to handle more complicated cases.
    old_names = _check_names(tensor)
    added_names = _check_names(index)
    idx = old_names.index(name)
    new_names = old_names[:idx] + added_names + old_names[idx + 1:]
    key = [slice(None) for _ in range(idx)] + [index.rename(None)]
    key = tuple(key)
    ret = tensor.rename(None)[key]
    return ret.refine_names(*new_names)


@deprecated(reason='Old helper functions.')
def gather(tensor: Tensor, index: Tensor) -> Tensor:
    if tensor.ndim != 2:
        raise NotImplementedError(f'tensor can only be a matrix, but got {tensor.ndim} dims.')
    if index.ndim != 1:
        raise NotImplementedError(f'index can only be a vector, but got {tensor.ndim} dims.')

    shared_name = index.names[0]
    index = index.align_as(tensor)
    dim = 1 if shared_name == tensor.names[0] else 0
    ret = tensor.rename(None).gather(dim, index.rename(None)).view(-1).refine_names(shared_name)
    return ret


@deprecated(reason='Old helper functions.')
def expand_as(tensor: Tensor, other: Tensor) -> Tensor:
    _check_names(tensor)
    other_names = _check_names(other)
    tensor = tensor.align_as(other)
    return tensor.rename(None).expand_as(other.rename(None)).refine_names(*other_names)


@deprecated(reason='Old helper functions.')
def get_named_range(size: int, name: str) -> Tensor:
    return get_range(size, 1, 0).refine_names(name)
