import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from functools import partial, update_wrapper, wraps
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch._C import Value
import torch.nn as nn

from dev_misc.trainlib import has_gpus

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
        dtype = type(x[0])
        if dtype is float:
            dtype = np.float32
        tensor = torch.from_numpy(np.asarray(x, dtype=dtype))
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


def get_zeros(*shape: int, cpu: bool = False) -> torch.Tensor:
    """Get zeros and use gpu if possible."""
    if has_gpus() and not cpu:
        return torch.cuda.FloatTensor(*shape).fill_(0.0)
    else:
        return torch.zeros(*shape)


def get_range(size: int, ndim: int, dim: int, cpu: bool = False):
    """Get torch.arange(size), and reshape it to have the shape where all `ndim` dimensions are of size 1 but the `dim` is `size`, and move it to GPU if possible."""
    shape = [1] * dim + [size] + [1] * (ndim - dim - 1)
    return get_tensor(torch.arange(size).long().reshape(*shape), cpu=cpu)


def pad_to_dense(a: List[np.ndarray], dtype=np.float32, pad_idx: int = 0, use_3d: bool = False, length_3d: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Modified from https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size.
    Return the sequences and paddings.
    '''
    maxlen = max(map(len, a))
    if use_3d:
        if length_3d is None:
            raise ValueError(f'Must pass `length_3d` if `use_3d` is `True`')
        seqs = np.zeros((len(a), maxlen, length_3d), dtype=dtype)
    else:
        seqs = np.zeros((len(a), maxlen), dtype=dtype)
    if pad_idx != 0:
        seqs.fill(pad_idx)
    paddings = np.zeros((len(a), maxlen), dtype=np.bool)
    for i, row in enumerate(a):
        seqs[i, :len(row)] = row
        paddings[i, :len(row)] = True
    return seqs, paddings


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
        value = getattr(self, attr)
        if value is not None and (_is_np_type(anno) or _is_tensor_type(anno)):
            try:
                shape = tuple(value.shape)
                out.append(f'{attr}: {shape}')
            except AttributeError:
                out.append(f'{attr}: {anno}')
        else:
            out.append(f'{attr}={value!r}')
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
    field_dict = dict()
    for field in fields(self):
        attr = field.name
        anno = field.type
        value = getattr(self, attr)
        if _is_tensor_type(anno):
            field_dict[attr] = value.detach().cpu().numpy()
        elif is_dataclass(value):
            field_dict[attr] = dataclass_numpy(value)
        else:
            field_dict[attr] = value
    return type(self)(**field_dict)


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
