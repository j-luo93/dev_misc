from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import torch
from prettytable import PrettyTable as pt
from typing_extensions import Protocol

# TODO(j_luo) Rename this to Stats maybe?
# TODO(j_luo) Add tests


def plain(value):
    '''Convert tensors or numpy arrays to one scalar.'''
    # Get to str, int or float first.
    if isinstance(value, torch.Tensor):
        assert value.numel() == 1
        value = value.item()
    elif isinstance(value, np.ndarray):
        assert value.size == 1
        value = value[0]
    # Format it nicely.
    if isinstance(value, (str, np.integer, int)):
        value = value
    elif isinstance(value, float):
        value = float(f'{value:.3f}')
    else:
        raise NotImplementedError
    return value


class SupportsStr(Protocol):

    def __str__(self) -> str: ...

# IDEA(j_luo) Maybe we can have a hierarchical representation of this. So that metrics that corresond to prf would be separate from metrics from losses.


class Metric:

    def __init__(self, name, value, weight=None, report_mean=True):
        self.name = name
        self._v = value
        self._w = weight
        self._report_mean = report_mean
        if report_mean and weight is None:
            raise ValueError(f'Must provide weight for reporting mean.')

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        if self.report_mean:
            return f'{plain(self._v)}/{plain(self._w)}={plain(self.mean)}'
        else:
            return f'{plain(self.total)}'

    def __repr__(self):
        return f'Metric(name={self.name}, value={self.value}, report_mean={self.report_mean})'

    def __eq__(self, other):
        return self.name == other.name

    def __add__(self, other):
        if isinstance(other, Metric):
            assert self == other, 'Cannot add two different metrics.'
            assert self.report_mean == other.report_mean
            if self.report_mean:
                return Metric(self.name, self._v + other._v, self._w + other._w, report_mean=self.report_mean)
            else:
                return Metric(self.name, self._v + other._v, report_mean=self.report_mean)
        else:
            # NOTE This is useful for sum() call.
            assert isinstance(other, (int, float)) and other == 0
            return self

    def __radd__(self, other):
        return self.__add__(other)

    def rename(self, name):
        '''This is in-place.'''
        self.name = name
        return self

    @property
    def value(self):
        return self._v

    @property
    def weight(self):
        return self._w if self.report_mean else 'N/A'

    @property
    def report_mean(self):
        return self._report_mean

    @property
    def mean(self):
        if self.report_mean:
            return self._v / (self._w + 1e-16)
        else:
            return 'N/A'

    @property
    def total(self):
        return self._v

    def clear(self):
        self._v = 0
        if self.report_mean:
            self._w = 0

    # TODO(j_luo) Think about api
    # FIXME(j_luo) This is misleading -- it's out-of-place.
    def with_prefix_(self, prefix: SupportsStr) -> Metric:
        return Metric(f'{prefix}_{self.name}', self._v, weight=self._w, report_mean=self._report_mean)


# TODO(j_luo) Add tests and simplify syntax.
# TODO(j_luo) metrics should be structured like dataclasses. If there are too many of them, we can place them under some catch-all value.
class Metrics:

    def __init__(self, *metrics):
        # Check all of metrics are of the same type. Either all str or all Metric.
        types = set([type(m) for m in metrics])
        assert len(types) <= 1

        if len(types) == 1:
            if types.pop() is str:
                self._metrics = {k: Metric(k, 0, 0) for k in keys}
            else:
                self._metrics = {metric.name: metric for metric in metrics}
        else:
            self._metrics = dict()

    def rename(self, old_name: str, new_name: str):
        '''This is in-place.'''
        metric = self._metrics[old_name]
        metric.rename(new_name)
        self._metrics[new_name] = metric
        del self._metrics[old_name]
        return self

    def items(self):
        yield from self._metrics.items()

    def __str__(self):
        out = '\n'.join([f'{k}: {m}' for k, m in self._metrics.items()])
        return out

    # IDEA(j_luo) Do sth similar to dataclass_repr with pprint?
    def __repr__(self):
        return f'Metrics({", ".join(self._metrics.keys())})'

    def __add__(self, other):
        if other is None:  # Allow `None + metrics`.
            return self
        if isinstance(other, Metric):
            other = Metrics(other)
        union_keys = set(self._metrics.keys()) | set(other._metrics.keys())
        metrics = list()
        for k in union_keys:
            m1 = self._metrics.get(k, 0)
            m2 = other._metrics.get(k, 0)
            metrics.append(m1 + m2)
        return Metrics(*metrics)

    def __radd__(self, other):
        return self.__add__(other)

    def __getattr__(self, attr: str):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f'Cannot find this attribute {attr}.')

    def __getitem__(self, key: str):
        return self._metrics[key]

    def get_table(self, title='', num_paddings: int = 0):
        """Return a pretty table based on this `Metrics` object, with an optional title.

        If `num_paddings` is non-zero, return a str object of this table instead with a specified number of leading whitespaces.
        """
        t = pt()
        if title:
            t.title = title
        t.field_names = 'name', 'value', 'weight', 'mean'
        for k in sorted(self._metrics.keys()):
            metric = self._metrics[k]
            t.add_row([k, plain(metric.value), plain(metric.weight), plain(metric.mean)])
        t.align = 'l'
        if num_paddings == 0:
            return t

        ret = ('\n' + str(t)).replace('\n', '\n' + ' ' * num_paddings)
        return ret

    def clear(self):
        self._metrics.clear()

    def __len__(self):
        return len(self._metrics)

    # TODO(j_luo) Think about api for this one.
    def with_prefix_(self, prefix: SupportsStr) -> Metrics:
        metrics = [metric.with_prefix_(prefix) for metric in self._metrics.values()]
        return Metrics(*metrics)

    def __iter__(self) -> Iterator[Metric]:
        yield from self._metrics.values()
