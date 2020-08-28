from __future__ import annotations

import inspect
from functools import wraps
from typing import Optional, Sequence, Union

import torch
from typeguard import typechecked

from dev_misc import LT

T = torch.Tensor


class NameMismatch(Exception):
    """Raise this if names mismatch."""


def elemwise(func):

    @wraps(func)
    @typechecked
    def wrapped(self: TensorX, other: Union[TensorX, float, int, float]) -> TensorX:
        if isinstance(other, TensorX):
            if set(self.names) != set(other.names):
                raise NameMismatch(f'Names mismatch for element-wise op {func}, got {self.names} and {other.names}.')
            ret = func(self.data, other.align_as(self).data)
        else:
            ret = func(self.data, other)
        return TensorX(ret, self.names)

    return wrapped


def inherit_prop(attr):

    @property
    @wraps(attr)
    def wrapped(self: TensorX):
        ret = attr.__get__(self.data)
        return ret

    return wrapped


def _check_names(tensor: T, names: Sequence[str]):
    if len(names) != tensor.ndim:
        raise ValueError(f'Expect equal values, but got {len(names)} and {tensor.ndim}.')
    if len(set(names)) < len(names):
        raise ValueError(f'Duplicate names in {names}.')


class Named:
    """This is used for torch.Tensor."""

    def __init__(self, tensor: T, names: Sequence[str]):
        _check_names(tensor, names)
        self.tensor = tensor
        self.names = names

    def __enter__(self):
        # Save old names if they exist.
        self.old_names = self.tensor.names
        self.tensor.rename_(*self.names)

    def __exit__(self, exc_type, exc_value, traceback):
        self.tensor.rename_(*self.old_names)


def _change_names(names: Sequence[str], old2new: Dict[str, str]):
    return tuple(old2new.get(old, old) for old in names)


class Renamed:
    """This is used for TensorX."""

    def __init__(self, tx: TensorX, old2new: Dict[str, str]):
        if not set(old2new) <= set(tx.names):
            raise ValueError(f'There are some unfound names in {old2new} for {tx.names}.')
        self.tx = tx
        self.old2new = old2new

    def __enter__(self):
        # Save old names if they exist.
        self.old_names = self.tx.names
        self.tx.names = _change_names(self.old_names, self.old2new)

    def __exit__(self, exc_type, exc_value, traceback):
        self.tx.names = self.old_names


class TensorX:

    def __init__(self, data: T, names: Sequence[str]):
        _check_names(data, names)
        self.data = data
        self.names = tuple(names)

    __mul__ = elemwise(T.__mul__)
    __add__ = elemwise(T.__add__)
    __radd__ = elemwise(T.__radd__)

    ndim = inherit_prop(T.ndim)
    shape = inherit_prop(T.shape)

    def __repr__(self):
        ret = f'Names: {self.names}'
        ret += '\n' + f'Sizes: {tuple(self.shape)}'
        ret += '\n' + repr(self.data)
        return ret

    @typechecked
    def select(self, name: str, index: int) -> TensorX:
        # HACK(j_luo): batch should be a reserved word?
        if name == 'batch':
            raise ValueError(f'__getitem__ should not be used on the batch dimension.')

        dim = self.names.index(name)
        new_names = self.names[:dim] + self.names[dim + 1:]
        return TensorX(self.data.select(dim, index), new_names)

    @typechecked
    def batched_select(self, name: str, index: TensorX) -> TensorX:
        if 'batch' in self.names:
            raise TypeError(f'Cannot use `batch_select` if tensor has the "batch" dimension.')

        if index.names != ('batch', ):
            raise TypeError(f'`index` should be a one-dimensional tensor with one name "batch", but got {index.names}.')

        dim = self.names.index(name)
        new_names = self.names[:dim] + ('batch', ) + self.names[dim + 1:]
        return TensorX(self.data.index_select(dim, index.data), new_names)

    @typechecked
    def align_as(self, other: TensorX) -> TensorX:
        with Named(self.data, self.names), Named(other.data, other.names):
            aligned_data = self.data.align_as(other.data).rename(None)
        return TensorX(aligned_data, other.names)

    def align_to(self, *names: str) -> TensorX:
        with Named(self.data, self.names):
            aligned_data = self.data.align_to(*names).rename(None)
        return TensorX(aligned_data, tuple(names))

    def softmax(self, name: str) -> TensorX:
        dim = self.names.index(name)
        return TensorX(self.data.softmax(dim=dim), self.names)

    def size(self, name: Optional[str] = None) -> Union[torch.Size, int]:
        if name is None:
            return self.data.size()
        dim = self.names.index(name)
        return self.data.size(dim=dim)

    def normalize_prob(self, name: str) -> TensorX:
        if (self.data < 0).any():
            raise TypeError(f'Cannot normalize this tensor since there are negative terms.')
        dim = self.names.index(name)
        return TensorX(self.data / self.data.sum(dim=dim, keepdim=True), self.names)

    @typechecked
    def contract(self, other: TensorX, name: str) -> TensorX:
        """This is only supported for two matrices with sizes:
        A: (m, n)
        B: (n, p)

        It sums (contracts) over the shared dimension, essentially equivalent to matmul.
        `name` is redundantly specified for readability.
        """
        if len(self.names) != 2 or len(other.names) != 2:
            raise TypeError(f'Can only support two matrices for now.')

        common_names = set(self.names) & set(other.names)
        if len(common_names) != 1:
            raise TypeError(f'Both matrices should have exactly one shared name.')

        common = common_names.pop()
        if common != name:
            raise ValueError(f'Specified name {name} does not match the shared name {common}.')
        m_name = [n for n in self.names if n != name][0]
        p_name = [n for n in other.names if n != name][0]

        with Named(self.data, self.names), Named(other.data, other.names):
            ret = self.data.align_to(m_name, name) @ other.data.align_to(name, p_name)
            ret.rename_(None)
        return TensorX(ret, [m_name, p_name])

    def rename_(self, old2new) -> TensorX:
        self.names = _change_names(self.names, old2new)
        return self
