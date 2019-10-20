import os

import numpy as np
import torch


def get_tensor(x):
    """Get a tensor from x, and move it to GPU if possible."""
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
    if os.environ.get('CUDA_VISIBLE_DEVICES', False):
        use_cuda = True
    else:
        use_cuda = False
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def get_zeros(*shape):
    """Get zeros and move to gpu if possible."""
    return get_tensor(torch.zeros(*shape))


def get_range(size, ndim, dim, name=None):
    """Get torch.arange(size), and reshape it to have the shape where all `ndim` dimensions are of size 1 but the `dim` is `size`, and move it to GPU if possible."""
    shape = [1] * dim + [size] + [1] * (ndim - dim - 1)
    names = [None] * dim + [name] + [None] * (ndim - dim - 1)
    return get_tensor(torch.arange(size).long().reshape(*shape).refine_names(*names))


def pad_to_dense(a, dtype=np.float32):
    '''
    Modified from https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size.
    '''
    maxlen = max(map(len, a))
    ret = np.zeros((len(a), maxlen), dtype=dtype)
    for i, row in enumerate(a):
        ret[i, :len(row)] += row
    return ret


def get_length_mask(lengths, max_len, dtype=np.bool):
    """Get a mask that is 1.0 if the index is less than the corresponding length."""
    if max_len < max(lengths):
        raise RuntimeError(f'max_len too small: {max_len} < {max(lengths)}.')
    mask = get_zeros(len(lengths), max_len)
    indices = get_range(max_len, 2, 1)
    mask[indices < get_tensor(lengths).view(-1, 1)] = 1.0
    return mask
