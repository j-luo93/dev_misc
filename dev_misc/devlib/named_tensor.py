import ctypes
import inspect
import logging
import warnings
from dataclasses import dataclass
from functools import wraps
from itertools import chain
from types import ModuleType
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Type, Union)

import torch
import torch.nn as nn

import dev_misc.devlib.helper as helper
from dev_misc import FT, LT
from dev_misc.utils import deprecated

Module = nn.Module
Tensor = torch.Tensor


# IDEA(j_luo) We can use an object that marks the size and the name of a dimension. That way, the name and the actual size are bound.

@dataclass
class Patch:
    module: ModuleType
    name: str  # NOTE(j_luo) This is the name of the patched function under module.
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

    def patch(self, patchable: _Patchable, create: bool = False, caller_name: Optional[str] = None):
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
            patch = Patch(patchable, name, unpatched, patched)

            _caller_name = caller_name if caller_name else name
            if _caller_name in self._patches:
                raise NameError(f'Duplicate name "{_caller_name}".')
            self._patches[_caller_name] = patch
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
            logging.debug(f'Named tensors already patched.')
            return
        for caller_name, patch in self._patches.items():
            setattr(patch.module, patch.name, patch.patched)
        self._patched = True

    def unpatch_named_tensors(self):
        if not self._patched:
            raise RuntimeError(f'Not patched yet.')
        for caller_name, patch in self._patches.items():
            setattr(patch.module, patch.name, patch.unpatched)
        self._patched = False


_patcher = _Patcher()
patch = _patcher.patch
patch_cls = _patcher.patch_cls
patch_named_tensors = _patcher.patch_named_tensors
unpatch_named_tensors = _patcher.unpatch_named_tensors
call_unpatched = _patcher.call_unpatched


@patch(torch.Tensor, create=True)
def hide_names(self):
    try:
        self._hidden_names_depth += 1
    except AttributeError:
        self._hidden_names_depth = 1

    if self.has_names() and not hasattr(self, '_hidden_names'):
        self._hidden_names = self.names
        self.rename_(None)
    return self


@patch(torch.Tensor, create=True)
def reveal_names(self):
    try:
        self._hidden_names_depth -= 1
    except AttributeError:
        self._hidden_names_depth = 0

    if self._hidden_names_depth == 0 and hasattr(self, '_hidden_names'):
        self.rename_(*self._hidden_names)
        del self._hidden_names
        del self._hidden_names_depth
    return self


def register_tensor_cls(cls):
    NoName.tensor_cls.add(cls)


class NoName:

    tensor_cls = set()

    def __init__(self, *args, **kwargs):
        all_args = chain(iter(args), iter(kwargs.values()))
        self._to_track = [arg for arg in all_args if torch.is_tensor(arg) or isinstance(arg, tuple(self.tensor_cls))]

    def __enter__(self):
        for tensor in self._to_track:
            tensor.hide_names()

    def __exit__(self, exc_type, exc_value, traceback):
        for tensor in self._to_track:
            tensor.reveal_names()


def drop_names(obj: object):
    """Drop names of all tensors nested in obj. Out-of-place."""

    def _drop_helper(obj):
        out = obj
        # Try mapping.
        if isinstance(obj, Mapping):
            out = {k: _drop_helper(value) for k, value in obj.items()}
        elif torch.is_tensor(obj):
            out = out.rename(None)
        elif not isinstance(obj, str):
            # Try iterator.
            try:
                out = type(obj)(_drop_helper(item) for item in obj)
            except TypeError:
                pass
        return out

    return _drop_helper(obj)


class Rename:

    def __init__(self, *tensors: torch.Tensor, **kwargs):
        self._to_track = tensors
        self._old2new = kwargs

    def __enter__(self):
        for tensor in self._to_track:
            tensor.rename_(**self._old2new)

    def __exit__(self, exc_type, exc_value, traceback):
        new2old = {v: k for k, v in self._old2new.items()}
        for tensor in self._to_track:
            tensor.rename_(**new2old)


class NameHelper:

    def __init__(self):
        self._reg = dict()

    def register_tensor(self, tensor: torch.Tensor):
        self._register_names(tensor)

    def _register_names(self, tensor: torch.Tensor):
        for size, name in zip(tensor.shape, tensor.names):
            if name is not None:
                if name in self._reg:
                    reg_size = self._reg[name]
                    # NOTE(j_luo) Ignore size 1 since it's used in broadcasting very often.
                    # IDEA(j_luo) But size 1 can actually be size 1, instead of being a dummy axis.
                    if reg_size != size and reg_size != 1 and size != 1:
                        raise RuntimeError(f'The same name is used for two different sizes other than 1.')
                    if reg_size == 1 and size != 1:
                        self._reg[name] = size
                else:
                    self._reg[name] = size

    @wraps(torch.Tensor.flatten)
    def flatten(self, tensor: torch.Tensor, names_to_flatten: Sequence[str], new_name: str) -> torch.Tensor:
        self._register_names(tensor)
        ret = tensor.flatten(names_to_flatten, new_name)
        self._register_names(ret)
        return ret

    @wraps(torch.Tensor.unflatten)
    def unflatten(self, tensor: torch.Tensor, name_to_unflatten: str, names: Sequence[str]) -> torch.Tensor:
        name_size_pairs = [(name, self._reg[name]) for name in names]
        ret = tensor.unflatten(name_to_unflatten, name_size_pairs)
        return ret


_Name = Union[str, Tuple[str, str]]
_Configuration = List[Tuple[_Patchable, List[_Name]]]
# Directly inherit the names.
_to_inherit: _Configuration = [
    (torch.nn.functional, ['leaky_relu', 'celu']),
    (torch, ['zeros_like', 'full_like', 'ones_like', 'layer_norm', 'where', 'min', 'max']),
    (torch.Tensor, ['addcmul_', 'addcdiv_', '__or__', '__and__', '__invert__', '__mod__', 'type_as', 'repeat'])
]
# Increase ref count for None.
_to_inc_refcount: _Configuration = [
    (torch.Tensor, ['refine_names', 'rename', 'rename_', 'align_to', 'align_as'])
]
# Drop names altogether.
_to_drop: _Configuration = [
    (torch.nn.Module, [('state_dict', 'module_state_dict')]),
    (torch.optim.Optimizer, [('state_dict', 'optimizer_state_dict')])
]
# Enhance the original api for reduction ops so that `dim` can be a name.
_to_reduce: _Configuration = [
    (torch.Tensor, ['any'])
]
_all_to_patch: Dict[str, _Configuration] = {
    'inherit': _to_inherit,
    'inc_refcount': _to_inc_refcount,
    'drop': _to_drop,
    'reduce': _to_reduce
}


def _gen_function(patchable: _Patchable, name: str, action: str, caller_name: str):
    old_func = getattr(patchable, name)

    if action == 'inherit':

        @wraps(old_func)
        def wrapped(tensor: Tensor, *args, **kwargs):

            with NoName(tensor, *args, **kwargs):
                ret = call_unpatched(tensor, *args, caller_name=caller_name, **kwargs)
            return ret.refine_names(*tensor.names)

    elif action == 'inc_refcount':

        _incref = ctypes.pythonapi.Py_IncRef
        _incref.argtypes = [ctypes.py_object]

        @wraps(old_func)
        def wrapped(*args, **kwargs):
            ret = call_unpatched(*args, caller_name=caller_name, **kwargs)
            for _ in range(2 * (len(args) + len(kwargs))):
                _incref(None)
            return ret

    elif action == 'drop':

        @wraps(old_func)
        def wrapped(*args, **kwargs):
            with NoName(*args, **kwargs):
                ret = call_unpatched(*args, caller_name=caller_name, **kwargs)
            return drop_names(ret)

    elif action == 'reduce':

        @wraps(old_func)
        def wrapped(tensor: Tensor, *args, dim: Union[int, str] = None, **kwargs):
            if dim is not None:
                names = tensor.names
                if isinstance(dim, str):
                    dim = names.index(dim)
                if kwargs.get('keepdims', False) or kwargs.get('keepdim', False):
                    new_names = names
                else:
                    new_names = names[:dim] + names[dim + 1:]
                with NoName(tensor, *args, **kwargs):
                    ret = call_unpatched(tensor, dim, *args, caller_name=caller_name, **kwargs)
                ret.rename_(*new_names)
            else:
                with NoName(tensor, *args, **kwargs):
                    ret = call_unpatched(tensor, *args, caller_name=caller_name, **kwargs)
            return ret

    return wrapped


# TODO(j_luo) This is side-effect.
for action, config in _all_to_patch.items():
    for patchable, names in config:
        for name in names:
            if isinstance(name, tuple):
                name, caller_name = name
            else:
                caller_name = name
            patched = _gen_function(patchable, name, action, caller_name)
            patch(patchable, caller_name=caller_name)(patched)


# @patch(torch.nn.Module)
# def state_dict(self, *args, **kwargs):
#     ret = call_unpatched(self, *args, **kwargs)
#     ret = {k: v.rename(None) for k, v in ret.items()}
#     return ret

# @patch(torch.optim.Optimizer)
# def state_dict(self, *args)


@patch(torch.nn.Module)
def _apply(self, func: Callable):

    @wraps(func)
    def wrapped(tensor: torch.Tensor):
        with NoName(tensor):
            ret = func(tensor)
        ret.rename_(*tensor.names)
        ret.hide_names()
        return ret

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
        # NOTE(j_luo) If `names` is provided, we use the new interface for named tensors.
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
        with NoName(*tensors):
            out = call_unpatched([tensor for tensor in tensors], dim=dim, out=out)
        return out.refine_names(*new_names)
    else:
        return call_unpatched(tensors, dim=dim, out=out)


@patch(torch)
def stack(tensors: Sequence[Tensor], dim: int = 0, out: Tensor = None, *, new_name: Optional[str] = None):
    if new_name is not None:
        tensor_names = [tensor.names for tensor in tensors]
        if any(tensor_names[0] != tn for tn in tensor_names[1:]):
            raise ValueError('Not all names are identical.')

        tensors = [tensor.align_as(tensors[0]) for tensor in tensors]
        with NoName(*tensors):
            out = call_unpatched(tensors, dim=-1, out=out)
        new_names = tensor_names[0] + (new_name, )
        return out.refine_names(*new_names)
    else:
        return call_unpatched(tensors, dim=dim, out=out)


Dim = Union[str, int]


class MustBeNamed(Exception):
    pass


@patch(torch.Tensor, create=True)
def has_full_names(self):
    return all(name is not None for name in self.names)


def _check_full_names(tensor: Tensor):
    if not tensor.has_full_names():
        raise MustBeNamed('Must be fully named.')


@patch(torch.Tensor)
def gather(self, dim: Dim, index: torch.LongTensor, *args, **kwargs):
    if isinstance(dim, int):
        with NoName(self, index):
            ret = call_unpatched(self, dim, index, *args, **kwargs)
        new_names = index.names
    elif isinstance(dim, str):
        _check_full_names(self)
        _check_full_names(index)

        common_names = [name for name in self.names if name != dim]
        index_unique_names = [name for name in index.names if name not in common_names]
        unique_name = None
        if len(index_unique_names) > 1:
            raise RuntimeError(f'Can only have at most one unique name for index.')
        elif len(index_unique_names) == 1:
            unique_name = index_unique_names[0]
            index = index.rename(**{unique_name: dim})
        # TODO(j_luo) unittest the following two lines.
        shape = [self.size(d) if d in common_names else -1 for d in self.names]
        index = index.align_as(self).expand(*shape)
        dim_int = self.names.index(dim)
        with NoName(self, index):
            ret = call_unpatched(self, dim_int, index, *args, **kwargs)
        if unique_name:
            index = index.rename(**{dim: unique_name})
        new_names = index.names
        if len(index_unique_names) == 0:
            ret = ret.squeeze(dim=dim_int)
            new_names = new_names[:dim_int] + new_names[dim_int + 1:]
    else:
        raise TypeError(f'Unsupported type for dim {type(dim)}.')
    return ret.refine_names(*new_names)


@patch(torch)
def embedding(weight: FT, index: LT, *args, **kwargs):
    with NoName(weight, index):
        out = call_unpatched(weight, index, *args, **kwargs)
    new_names = index.names + weight.names[-1:]
    return out.rename_(*new_names)


@patch(torch.Tensor)
def expand_as(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if self.has_names() and other.has_names():
        obj = self.align_as(other)
        with NoName(obj, other):
            out = call_unpatched(obj, other)
        out.rename_(*obj.names)
        return out
    else:
        with NoName(self, other):
            return call_unpatched(self, other)


@patch(torch.nn.functional)
def cross_entropy(tensor: FT, target: LT, weight: Optional[FT] = None, **kwargs):
    with NoName(tensor, target, weight):
        out = call_unpatched(tensor, target, weight, **kwargs)
    return out


@patch(torch)
def topk(tensor: FT, k: int, dim: Dim = -1, *args, **kwargs):
    if isinstance(dim, int):
        return call_unpatched(tensor, k, dim=dim, *args, **kwargs)
    else:
        dim_int = tensor.names.index(dim)
        with NoName(tensor):
            values, indices = call_unpatched(tensor, k, dim=dim_int, *args, **kwargs)
        values.rename_(*tensor.names)
        # FIXME(j_luo) If each name is associated with a fixed value, then inheriting the name is wrong.
        indices.rename_(*tensor.names)
        return values, indices


@patch(torch)
def repeat_interleave(input_, repeats, dim=None):
    dim_int = dim
    if isinstance(dim, str):
        dim_int = input_.names.index(dim)

    with NoName(input_, repeats):
        ret = call_unpatched(input_, repeats, dim=dim_int)

    if isinstance(dim, str):
        new_names = input_.names[:dim_int] + (dim, ) + input_.names[dim_int + 1:]
        ret = ret.refine_names(*new_names)

    return ret


def duplicate(tensor: torch.Tensor, name: str, size: int, dup_name: str):
    assert tensor.has_names()
    names = tensor.names
    sizes = tensor.shape
    dim = names.index(name)
    new_sizes = sizes[: dim + 1] + (size, ) + sizes[dim + 1:]
    new_names = names[: dim + 1] + (dup_name, ) + names[dim + 1:]
    ret = tensor.align_to(*new_names)
    return ret.expand(*new_sizes)


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


patch_cls(torch.nn, 'Module', NamedModule)
patch_cls(torch.nn, 'Linear', NamedModule)
patch_cls(torch.nn, 'Embedding', NamedModule)
patch_cls(torch.nn, 'LayerNorm', NamedModule)


def get_named_range(size: int, name: str) -> Tensor:
    return helper.get_range(size, 1, 0).refine_names(name)
