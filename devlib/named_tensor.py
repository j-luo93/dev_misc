from typing import List, Tuple

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


def self_attend(mod: Module, tensor: Tensor) -> Tuple[Tensor, Tensor]:
    names = _safe_get_name(tensor)
    name_len, name_batch = names[:2]
    name_len_query = f'{name_len}_query'
    name_len_key = f'{name_len}_key'
    tensor = tensor.rename(None)
    output, weight = mod(tensor, tensor, tensor)
    output = output.refine_names(*names)
    weight = weight.refine_names(name_batch, name_len_query, name_len_key)
    return output, weight
