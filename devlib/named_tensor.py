from typing import Tuple

import torch
import torch.nn as nn

Module = nn.Module
Tensor = torch.Tensor


class MustBeNamed(Exception):
    pass


def _safe_get_name(tensor: Tensor) -> bool:
    """Check whether tensor is named."""
    if tensor.names is None:
        raise MustBeNamed('Tensor must be named.')
    return tensor.names


def embed(mod: Module, tensor: Tensor, new_dim_name: str) -> Tensor:
    """Embed a tensor and adjust the names."""
    names = _safe_get_name(tensor)
    new_names = names + (new_dim_name, )
    return mod(tensor.rename(None)).refine_names(*new_names)


def collapse(tensor: Tensor, *names: Tuple[str]) -> Tensor:
    """Collapse a span of named dimensions into one. The collapsed dimension will joined by 'X'."""
    names = set(names)
    old_names = _safe_get_name(tensor)
    dims = sorted([i for i, name in enumerate(old_names) if name in names])
    start_dim = min(dims)
    end_dim = max(dims)
    if end_dim - start_dim != len(dims) - 1:
        raise ValueError(
            f'Names are not contiguous! Tensor has names {old_names} and we are trying to collapse {names}.')

    mid_name = 'X'.join(old_names[start_dim: end_dim + 1])
    new_names = old_names[:start_dim] + (mid_name, ) + old_names[end_dim + 1:]

    shape = tensor.shape
    mid_dim = 1
    for dim in shape[start_dim: end_dim + 1]:
        mid_dim *= dim
    new_shape = shape[:start_dim] + (mid_dim, ) + shape[end_dim + 1:]
    return tensor.rename(None).view(*new_shape).refine_names(*new_names)
