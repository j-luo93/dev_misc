from typing import Dict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Module = nn.Module
Tensor = torch.Tensor


class MustBeNamed(Exception):
    pass


def _safe_get_name(tensor: Tensor) -> bool:
    """Check whether tensor is named."""
    if all([name is None for name in tensor.names]):
        raise MustBeNamed('Tensor must be named.')
    return tensor.names

# TODO(j_luo) Document name changes for the following helper functions.


def leaky_relu(tensor: Tensor, **kwargs) -> Tensor:
    names = _safe_get_name(tensor)
    return F.leaky_relu(tensor.rename(None), **kwargs).refine_names(*names)


def embed(mod: Module, tensor: Tensor, new_dim_name: str) -> Tensor:
    """Embed a tensor and adjust the names."""
    names = _safe_get_name(tensor)
    new_names = names + (new_dim_name, )
    return mod(tensor.rename(None)).refine_names(*new_names)


def self_attend(mod: Module, tensor: Tensor, new_name: str) -> Tuple[Tensor, Tensor]:
    old_names = _safe_get_name(tensor)
    new_names = old_names[:-1] + (new_name,)

    name_len, name_batch = old_names[:2]
    name_len_T = f'{name_len}_T'
    tensor = tensor.rename(None)
    output, weight = mod(tensor, tensor, tensor)
    output = output.refine_names(*new_names)
    weight = weight.refine_names(name_batch, name_len, name_len_T)
    return output, weight


def adv_index(tensor: Tensor, name: str, index: Tensor) -> Tensor:
    # TODO(j_luo) Expand this function to handle more complicated cases.
    old_names = _safe_get_name(tensor)
    added_names = _safe_get_name(index)
    idx = old_names.index(name)
    new_names = old_names[:idx] + added_names + old_names[idx + 1:]
    key = [slice(None) for _ in range(idx)] + [index.rename(None)]
    key = tuple(key)
    ret = tensor.rename(None)[key]
    return ret.refine_names(*new_names)


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
