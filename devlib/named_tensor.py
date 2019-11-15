import ctypes
import inspect
import warnings
from dataclasses import dataclass
from functools import wraps
from itertools import chain
from types import ModuleType
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import torch
import torch.nn as nn
from deprecated import deprecated

import devlib.helper as helper

Module = nn.Module
Tensor = torch.Tensor


@dataclass
class Patch:
    module: ModuleType
    unpatched: Optional[Callable]
    patched: Callable


def _get_caller_name(stacklevel=1):
    frame = inspect.currentframe()
    while stacklevel > 0:
        frame = frame.f_back
        stacklevel -= 1
    return frame.f_code.co_name


_Patchable = Union[ModuleType, Type]
_Patched = Union[Type, Callable]


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

    def patch(self, patchable: _Patchable, create: bool = False):
        """
        Note that this doesn't actually patch the functions -- it just hooks them. To actually patch them, use `patch_named_tensors`.
        If `create` is set to True, create a new method instead.
        """

        def decorator(patched: _Patched):
            name = patched.__name__
            if create:
                if hasattr(patchable, name):
                    raise NameError(
                        f'A method/function named "{name}" in {patchable} already exists. Cannot create a new one for it.')
                unpatched = None
            else:
                unpatched = getattr(patchable, name)
            patch = Patch(patchable, unpatched, patched)
            if name in self._patches:
                raise NameError(f'Duplicate name "{name}".')
            self._patches[name] = patch
            return patched

        return decorator

    def patch_cls(self, module: ModuleType, cls_name: Type, base_cls: Type):
        """Patch a class. This works by defining a subclass of `parent_cls` (the class named `cls_name` in `module`) with a new `base_cls` as the first parent class."""
        parent_cls = getattr(module, cls_name)
        new_cls = type(cls_name, (base_cls, parent_cls), dict())
        self.patch(module)(new_cls)

    def call_unpatched(self, *args, stacklevel: int = 1, caller_name: str = None, **kwargs):
        if caller_name is None:
            caller_name = _get_caller_name(stacklevel=stacklevel + 1)
        patch = self._patches[caller_name]
        unpatched = patch.unpatched
        if unpatched is None:
            raise RuntimeError(f'You should not have called this function -- this is created, not patched.')
        return unpatched(*args, **kwargs)

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
patch_cls = _patcher.patch_cls
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


@patch(torch.Tensor, create=True)
def hide_names(self):
    if self.has_names() and not hasattr(self, '_hidden_names'):
        self._hidden_names = self.names
        self.rename_(None)
    return self


@patch(torch.Tensor, create=True)
def reveal_names(self):
    if hasattr(self, '_hidden_names'):
        self.rename_(*self._hidden_names)
        del self._hidden_names
    return self


class NoName:

    def __init__(self, *args, **kwargs):
        all_args = chain(iter(args), iter(kwargs.values()))
        self._to_track = list(filter(torch.is_tensor, all_args))

    def __enter__(self):
        for tensor in self._to_track:
            tensor.hide_names()

    def __exit__(self, exc_type, exc_value, traceback):
        for tensor in self._to_track:
            tensor.reveal_names()


_to_inherit: List[Tuple[_Patchable, List[str]]] = [
    (torch.nn.functional, ['leaky_relu']),
    (torch, ['zeros_like']),
    (torch.Tensor, ['addcmul_', 'addcdiv_'])
]


def _gen_function(patchable: _Patchable, name: str):
    old_func = getattr(patchable, name)

    @wraps(old_func)
    def wrapped(tensor: Tensor, *args, **kwargs):
        with NoName(tensor, *args, **kwargs):
            ret = call_unpatched(tensor, *args, caller_name=name, **kwargs)
        return ret.refine_names(*tensor.names)

    return wrapped


# TODO(j_luo) This is side-effect.
for patchable, names in _to_inherit:
    for name in names:
        patched = _gen_function(patchable, name)
        patch(patchable)(patched)


@patch(torch.nn.Module)
def state_dict(self, *args, **kwargs):
    ret = call_unpatched(self, *args, **kwargs)
    ret = {k: v.rename(None) for k, v in ret.items()}
    return ret


@patch(torch.nn.Module)
def _apply(self, func: Callable):

    @wraps(func)
    def wrapped(tensor: torch.Tensor):
        tensor.hide_names()
        return func(tensor)

    def post_apply(module: torch.nn.Module):
        for child in module.children():
            post_apply(child)

        for key, param in module._parameters.items():
            if param is not None:
                param.reveal_names()
                if param.grad is not None:
                    param.grad.reveal_names()

        for key, buf in module._buffers.items():
            if buf is not None:
                buf.reveal_names()
        return module

    # This process is broken down into two, so that the first part can still be safely used. The second part is hacked because it should be fine to change attributes of tensors.
    ret = call_unpatched(self, wrapped)
    post_apply(ret)

    return ret


NameType = Union[Sequence[str], str]


@patch(torch)
def cat(tensors: Sequence[Tensor], dim: int = 0, out: Tensor = None, *, names: Optional[NameType] = None, new_name: Optional[str] = None):
    if names is not None:
        # NOTE(j_luo) If both arguments are provided, we use the new interface for named tensors.
        if isinstance(names, str):
            names = [names] * len(tensors)
        if len(names) != len(tensors):
            raise ValueError(f'Mismatched lengths for names and tensors.')
        dims = [tensor.names.index(name) for tensor, name in zip(tensors, names)]
        if any(dims[0] != dim for dim in dims[1:]):
            # IDEA(j_luo) Maybe we can add automatic re-alignment.
            raise ValueError(f'Not all names are in the same dimension.')
        dim = dims[0]

        remaining_names = [tensor.names[:dim] + tensor.names[dim + 1:] for tensor in tensors]
        if any(remaining_names[0] != rn for rn in remaining_names[1:]):
            raise ValueError(f'Not all remaining renames are matched.')

        new_names = tensors[0].names[:dim] + (new_name, ) + tensors[0].names[dim + 1:]
        out = call_unpatched([tensor.rename(None) for tensor in tensors], dim=dim, out=out)
        return out.refine_names(*new_names)
    else:
        return call_unpatched(tensors, dim=dim, out=out)


class NamedModule:

    def _refine_names_helper(self, attr_path: List[str], names: Sequence[str]):
        if len(attr_path) == 0:
            raise RuntimeError(f'Path to the attribute has length 0.')
        elif len(attr_path) == 1:
            tensor_name = attr_path[0]
            tensor = getattr(self, tensor_name)
            named_tensor = tensor.refine_names(*names)
            # NOTE(j_luo) `refine_names` is out-of-place. To make sure that the same parameter/tensor is tracked (e.g., by an optimizer),
            # we can use `rename_` instead.
            tensor.rename_(*named_tensor.names)
        else:
            child_module = getattr(self, attr_path[0])
            child_module._refine_names_helper(attr_path[1:], *names)

    def refine_names(self, attr_name: str, names: Sequence[str]):
        dot_separated = attr_name.split('.')
        self._refine_names_helper(dot_separated, names)


patch_cls(torch.nn, 'Linear', NamedModule)

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
    return helper.get_range(size, 1, 0).refine_names(name)
