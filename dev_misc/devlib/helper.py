import re
import logging
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import partial, update_wrapper, wraps
from typing import Any, List, Sequence, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn

# import devlib.named_tensor as named_tensor

# IDEA(j_luo) Tensor generic type!
Tensor = torch.Tensor
Tensorable = Union[np.ndarray, Tensor, List]


def get_tensor(x: Tensorable, cpu=False) -> Tensor:
    """Get a tensor from x, and move it to GPU if possible (unless `cpu` is True)."""
    # Convert it to tensor.
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    elif torch.is_tensor(x):
        tensor = x
    elif isinstance(x, (list, tuple)):
        tensor = torch.from_numpy(np.asarray(x))
    else:
        raise NotImplementedError(f'Unsupported type {type(x)}.')
    # Check if we need to use cuda.
    if os.environ.get('CUDA_VISIBLE_DEVICES', False) and not cpu:
        use_cuda = True
    else:
        use_cuda = False
    if use_cuda:
        tensor = tensor.cuda()
    else:
        tensor = tensor.cpu()
    # # NOTE(j_luo) Convert everything back to unnamed tensor so that we don't have to deal with unsupported ops on named tensors later.
    # tensor.rename_(None)  # TODO(j_luo) use something from named_tensor.py?
    return tensor


def get_zeros(*shape: int, cpu: bool = False):
    """Get zeros and move to gpu if possible."""
    return get_tensor(torch.zeros(*shape), cpu=cpu)


def get_range(size: int, ndim: int, dim: int, cpu: bool = False):
    """Get torch.arange(size), and reshape it to have the shape where all `ndim` dimensions are of size 1 but the `dim` is `size`, and move it to GPU if possible."""
    shape = [1] * dim + [size] + [1] * (ndim - dim - 1)
    return get_tensor(torch.arange(size).long().reshape(*shape), cpu=cpu)


def pad_to_dense(a: List[np.ndarray], dtype=np.float32) -> np.ndarray:
    '''
    Modified from https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size.
    '''
    maxlen = max(map(len, a))
    ret = np.zeros((len(a), maxlen), dtype=dtype)
    for i, row in enumerate(a):
        ret[i, :len(row)] += row
    return ret


def get_array(array_like: Sequence[Any]) -> np.ndarray:
    """
    Create an array of arbitrary objects without converting them into multidimensional numpy arrays.
    See https://stackoverflow.com/questions/38774922/prevent-numpy-from-creating-a-multidimensional-array
    """
    arr = np.empty(len(array_like), dtype=object)
    arr[:] = array_like
    return arr


def get_length_mask(lengths: Tensor, max_len: int, cpu: bool = False) -> torch.BoolTensor:
    """Get a mask that is True if the index is less than the corresponding length."""
    if lengths.ndim != 1:
        raise TypeError(f'Expect lengths to be a vector, but got {lengths.ndim} dimensions.')
    mask = get_zeros(len(lengths), max_len, cpu=cpu).bool()
    indices = get_range(max_len, 2, 1, cpu=cpu)
    within_length = indices < get_tensor(lengths, cpu=cpu).unsqueeze(dim=-1)
    mask[within_length] = True
    return mask


def _is_tensor_type(x: Any) -> bool:
    # FIXME(j_luo) This is extremely hacky. Need typing for torch tensor types. This would break down if future annotations are used.
    attr = '__name__' if hasattr(x, '__name__') else '_name'
    ret = False
    if hasattr(x, attr):
        value = getattr(x, attr)
        ret = ret | (value is not None and 'Tensor' in getattr(x, attr))
    if hasattr(x, '__args__'):
        for type_ in x.__args__:
            ret = ret | _is_tensor_type(type_)
    return ret


def _is_np_type(x: Any) -> bool:
    # FIXME(j_luo) This is extremely hacky. Need typing for numpy array types.
    ret = False
    if x is np.ndarray:
        return True
    if hasattr(x, '__args__'):
        for type_ in x.__args__:
            ret = ret | _is_np_type(type_)
    return ret


def dataclass_size_repr(self):
    """ __repr__ for dataclasses so that bt can generate repr result for tensors or arrays with only shape information printed out."""
    # TODO(j_luo) also print out names?
    # TODO(j_luo) should inheirt old __repr__ so that some fields with repr == False are taken care of.
    out = list()
    for field in fields(self):
        attr = field.name
        anno = field.type
        if _is_np_type(anno) or _is_tensor_type(anno):
            shape = tuple(getattr(self, attr).shape)
            out.append(f'{attr}: {shape}')
        else:
            out.append(f'{attr}={getattr(self, attr)!r}')
    cls = type(self)
    header = cls.__name__
    out = [re.sub(r'(\n\t*)', r'\1\t', o) for o in out]
    content = ',\n\t'.join(out)
    return f'{header}(\n\t{content}\n)'


T = TypeVar('T')


def dataclass_cuda(self: T) -> T:
    """Move tensors to gpu if possible. This is in-place."""
    for field in fields(self):
        attr = field.name
        value = getattr(self, attr)
        if torch.is_tensor(value):
            # TODO(j_luo) use something from named_tensor.py?
            names = value.names
            setattr(self, attr, get_tensor(value).refine_names(*names))
    return self


def dataclass_numpy(self: T) -> T:
    """Convert tensors to numpy arrays if possible. This is out-of-place."""
    ret = deepcopy(self)
    for attr, field in ret.__dataclass_fields__.items():
        anno = field.type
        if _is_tensor_type(anno):
            tensor = getattr(ret, attr)
            setattr(ret, attr, tensor.cpu().numpy())
    return ret


# NOTE(j_luo) Batch dataclasses will inherit the customized __repr__. Not sure if he type hint is correct, but it helps vscode find the function signature.
batch_class: Type[dataclass] = update_wrapper(partial(dataclass, repr=False), dataclass)


@batch_class
class BaseBatch:
    __repr__ = dataclass_size_repr
    cuda = dataclass_cuda
    numpy = dataclass_numpy


def debug_stats(message: str, tensor: Tensor):
    logging.info(f'{message} nir:')
    logging.info(f'\tshape/device:\t{tensor.shape}/{tensor.device}')
    logging.info('\tNAN:\t' + str(torch.isnan(tensor).any().item()))
    if not tensor.dtype is torch.bool:
        logging.info('\tINF:\t' + str(torch.isinf(tensor).any().item()))
    logging.info(f'\tmax/min:\t{tensor.max().item()}/{tensor.min().item()}')
