import os

import numpy as np
import torch

def get_tensor(data, dtype=None, requires_grad=False, sparse=False, use_cuda=True):
    use_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', False) and use_cuda # NOTE only use cuda when it's not overriden and there is a device available

    if dtype is None: # NOTE infer dtype
        dtype = 'f'
        if isinstance(data, np.ndarray) and issubclass(data.dtype.type, np.integer):
            dtype = 'l'

    # NOTE directly declare data. I believe it's faster on cuda, although I'm not entirely sure
    requires_grad = requires_grad and dtype == 'f'
    assert dtype in ['f', 'l']
    if use_cuda:
        module = getattr(torch, 'cuda')
    else:
        module = torch
    if dtype == 'f':
        cls = getattr(module, 'FloatTensor')
        dtype = 'float32'
    elif dtype == 'l':
        cls = getattr(module, 'LongTensor')
        dtype = 'int64'
    ret = cls(np.asarray(data, dtype=dtype))
    ret.requires_grad = requires_grad
    return ret

def get_zeros(*shape, **kwargs):
    if len(shape) == 1 and isinstance(shape[0], torch.Size): # NOTE deal with 1D tensor whose shape cannot be unpacked
        shape = list(shape[0])
    return get_tensor(np.zeros(shape), **kwargs)

def get_eye(n):
    return get_tensor(np.eye(n))
