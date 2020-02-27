"""A `dataclass`-like construct that helps generate different bash commands based on a grid-like configuration.

The configuration is represented by a `.grid` file, which has the following syntax for each line:
`full_key[,short_key]:type=value[,another_value,...]`
Extra space is allowed between fields not but inside fields.
`type` can be either `indep`, an independent value or `dep` a dependent value.
If it's `dep`, the expression for computing its value should be specified, and the key for the value that it is dependent on should be specified in braces (full key or short).
The entire expression should be encapsulated into a f-string for evaluation.
For instance:
```
pi: indep = 3.14
radius, r: indep = 1, 3, 5
area, a: dep = ({r} ** 2) * {pi}
```
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple,
                    TypeVar, Union)

line_pat = re.compile(r'^(\w+)(?:(?:,\s*)(\w+))?\s*:\s*(indep|dep)\s*=\s*(.+)$')


class FormatError(Exception):
    """Raise this error when any line doesn't match `line_pat` pattern or doesn't meet certain expectations."""


V = TypeVar('V', int, float, str)


class Graph:
    """Based on https://www.geeksforgeeks.org/topological-sorting/."""

    def __init__(self):
        self.graph = defaultdict(set)  # dictionary containing adjacency List
        self.V = set()

    def add_vertice(self, vertice):
        self.V.add(vertice)

    # function to add an edge to graph
    def add_edge(self, u, v):
        self.graph[u].add(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited.add(v)

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if i not in visited:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = set()
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for v in self.V:
            if v not in visited:
                self.topologicalSortUtil(v, visited, stack)

        # Return contents of the stack
        return stack


@dataclass(frozen=True)
class Field:
    full_key: str


@dataclass(frozen=True)
class Indep(Field):
    values: FrozenSet[V]
    short_key: Optional[str] = ''

    def __repr__(self):
        return f'Indep({self.full_key})'


@dataclass(frozen=True)
class Dep(Field):
    value: str
    dep_fields: FrozenSet[Field]
    short_key: Optional[str] = ''

    def get_all_indep(self) -> Set[Indep]:
        ret = set()
        for dep in self.dep_fields:
            if isinstance(dep, Indep):
                ret.add(dep)
            else:
                ret.update(dep._get_all_indep())
        return ret

    def __repr__(self):
        indeps = self.get_all_indep()
        return f'Dep({self.full_key}) <- {indeps}'


dep_pat = re.compile(r'\{(\w+)\}')


class FieldFactory:

    _instances: ClassVar[Dict[str, Field]] = dict()

    def get_field(self, key: str) -> Field:
        cls = type(self)
        return cls._instances[key]

    def create_field(self, type_: str, full_key: str, values: Union[str, List[V]], short_key: Optional[str] = '') -> Field:
        cls = type(self)
        if type_ not in ['indep', 'dep']:
            raise FormatError(f'Unexcepted type value {type_}.')
        if full_key in cls._instances:
            raise FormatError(f'Duplicate full key {full_key}.')
        if short_key and short_key in cls._instances:
            raise FormatError(f'Duplicate short key {short_key}.')
        if type_ == 'indep':
            field = Indep(full_key, tuple(values), short_key=short_key)
        else:
            value = values[0]
            dep_keys = set(dep_pat.findall(value))
            dep_fields = frozenset(cls._instances[key] for key in dep_keys)
            field = Dep(full_key, value, dep_fields, short_key=short_key)

        cls._instances[full_key] = field
        if short_key:
            cls._instances[short_key] = field
        return field

    def _get_all_typed_fields(self, type_: str) -> List[Field]:
        cls = type(self)
        field_cls = Indep if type_ == 'indep' else Dep
        return sorted(filter(lambda field: isinstance(field, field_cls), cls._instances.values()), key=lambda field: field.full_key)

    def get_all_indep(self) -> List[Indep]:
        return self._get_all_typed_fields('indep')

    def get_all_dep(self) -> List[Dep]:
        return self._get_all_typed_fields('dep')


class Grid:

    def __init__(self, file_path: Path):
        ff = FieldFactory()

        def convert_values(values):
            try:
                values = [int(v) for v in values]
            except ValueError:
                try:
                    values = [float(v) for v in values]
                except ValueError:
                    pass
            return values

        with file_path.open('r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip()
                match = line_pat.match(line)
                if match is None:
                    raise FormatError(f"'{line}' doesn't match line_pat pattern.")
                full_key = match.group(1)
                short_key = match.group(2)
                type_ = match.group(3)
                values = convert_values(match.group(4).split(','))

                if type_ == 'dep' and len(values) > 1:
                    raise FormatError(f'Expect only one expression for dependent values.')

                ff.create_field(type_, full_key, values, short_key)

        g = Graph()
        for f in ff.get_all_indep():
            g.add_vertice(f)
        for f in ff.get_all_dep():
            g.add_vertice(f)
            for dep in f.dep_fields:
                g.add_edge(dep, f)
        self.stack = g.topologicalSort()

    def generate(self):

        value_dict = dict()
        full_keys = set(field.full_key for field in self.stack)
        ret = list()

        def update(field, v):
            value_dict[field.full_key] = v
            if field.short_key:
                value_dict[field.short_key] = v

        def helper(ind: int):
            if ind >= len(self.stack):
                ret.append(' '.join(f'--{k} {v}' for k, v in sorted(value_dict.items()) if k in full_keys))
                return
            field = self.stack[ind]
            if isinstance(field, Indep):
                for v in field.values:
                    update(field, v)
                    helper(ind + 1)
            else:
                replaced = eval(f'f"{field.value}"', {}, value_dict)
                v = eval(replaced)
                update(field, v)
                helper(ind + 1)

        helper(0)
        return ret


if __name__ == "__main__":
    file_path = Path(sys.argv[1])
    grid = Grid(file_path)
    for setting in grid.generate():
        print(setting)
