import os
from functools import wraps
from typing import List, Union

import numpy as np
import torch

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
    elif isinstance(x, list):
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
    # NOTE(j_luo) Convert everything back to unnamed tensor so that we don't have to deal with unsupported ops on named tensors later.
    tensor.rename_(None)  # TODO(j_luo) use something from named_tensor.py?
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


def get_length_mask(lengths: Tensor, max_len: int, cpu: bool = False) -> torch.BoolTensor:
    """Get a mask that is True if the index is less than the corresponding length."""
    if lengths.ndim != 1:
        raise TypeError(f'Expect lengths to be a vector, but got {lengths.ndim} dimensions.')
    mask = get_zeros(len(lengths), max_len, cpu=cpu).bool()
    indices = get_range(max_len, 2, 1, cpu=cpu)
    within_length = indices < get_tensor(lengths, cpu=cpu).unsqueeze(dim=-1)
    # TODO(j_luo) ugly
    mask[within_length] = True
    return mask


def dataclass_size_repr(self):
    """ __repr__ for dataclasses so that bt can generate repr result for tensors or arrays with only shape information printed out."""
    # TODO(j_luo) also print out names?
    # TODO(j_luo) should inheirt old __repr__ so that some fields with repr == False are taken care of.
    out = list()
    for attr, field in self.__dataclass_fields__.items():
        anno = field.type
        # IDEA(j_luo) need typing for torch tensor types.
        if anno is np.ndarray or 'Tensor' in anno.__name__:
            shape = tuple(getattr(self, attr).shape)
            out.append(f'{attr}: {shape}')
        else:
            out.append(f'{attr}={getattr(self, attr)!r}')
    cls = type(self)
    return f"{cls.__name__}({', '.join(out)})"


def dataclass_cuda(self):
    """Move tensors to gpu if possible."""
    for attr, field in self.__dataclass_fields__.items():
        anno = field.type
        if 'Tensor' in anno.__name__:
            tensor = getattr(self, attr)
            names = tensor.names
            # TODO(j_luo) use something from named_tensor.py?
            setattr(self, attr, get_tensor(tensor).refine_names(*names))
