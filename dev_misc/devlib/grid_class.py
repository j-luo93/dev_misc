"""A `dataclass`-like construct that helps generate different bash commands based on a grid-like configuration.

The configuration is represented by a `.grid` file, which has the following syntax for each line:
`full_key[,short_key]:type=value[,another_value,...]`
Extra space is allowed between fields not but inside fields.
`type` can be one of the following values:
    1. `indep`, an independent value,
    2. `dep`, a dependent value,
    3. `count`, an independent value but will not be part of the arguments, and you have specify at most one of them,
    4. `flag`, just a flag. In this case, no value should be provided.
    5. `header`, this is prepended to every command in the original specified order.
If it's `dep`, the expression for computing its value should be specified, and the key for the value that it is dependent on should be specified in braces (full key or short).
The entire expression should be encapsulated into a f-string for evaluation.
For instance:
```
pi: indep = 3.14
radius, r: indep = 1, 3, 5
area, a: dep = ({r} ** 2) * {pi}
```

There are also some reserved keywords to trigger specifal behaviors. For instance, `matching_log_dir` should generate matching log directories automatically.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple,
                    TypeVar, Union)

nonflag_pat = re.compile(r'^(\w+)(?:(?:,\s*)(\w+))?\s*:\s*(indep|dep|count|header)\s*=\s*(.+)$')
flag_pat = re.compile(r'^(\w+)(?:(?:,\s*)(\w+))?\s*:\s*(flag)\s*$')


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
class Count(Indep):
    """`Count` is a subclass of `Indep`, critical for `isinstance` check."""

    def __repr__(self):
        return f'Count({self.full_key})'


@dataclass(frozen=True)
class Flag(Field):
    short_key: Optional[str] = ''

    def __repr__(self):
        return f'Flag({self.full_key})'


@dataclass(frozen=True)
class Header(Field):
    value: V
    short_key: Optional[str] = ''

    def __repr__(self):
        return f'Header({self.full_key})'


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


class MoreThanOneCountProvided(Exception):
    """Raise this if more than one `count` is provided."""


class FieldFactory:

    _instances: ClassVar[Dict[str, Field]] = dict()
    _count: ClassVar[Count] = None

    def get_field(self, key: str) -> Field:
        cls = type(self)
        return cls._instances[key]

    @property
    def count(self) -> Optional[Count]:
        cls = type(self)
        return cls._count

    def create_field(self, type_: str, full_key: str, short_key: Optional[str] = '', values: Optional[Union[str, List[V]]] = None) -> Field:
        cls = type(self)
        if type_ not in ['indep', 'dep', 'flag', 'count', 'header']:
            raise FormatError(f'Unexcepted type value {type_}.')
        if full_key in cls._instances:
            raise FormatError(f'Duplicate full key {full_key}.')
        if short_key and short_key in cls._instances:
            raise FormatError(f'Duplicate short key {short_key}.')

        if type_ == 'indep':
            field = Indep(full_key, tuple(values), short_key=short_key)
        elif type_ == 'dep':
            value = values[0]
            dep_keys = set(dep_pat.findall(value))
            dep_fields = frozenset(cls._instances[key] for key in dep_keys)
            field = Dep(full_key, value, dep_fields, short_key=short_key)
        elif type_ == 'count':
            field = Count(full_key, tuple(values), short_key=short_key)
            if cls._count is not None:
                raise MoreThanOneCountProvided()
            cls._count = field
        elif type_ == 'header':
            field = Header(full_key, values[0], short_key=short_key)
        else:
            field = Flag(full_key, short_key=short_key)

        cls._instances[full_key] = field
        if short_key:
            cls._instances[short_key] = field
        return field

    def _get_all_typed_fields(self, field_cls) -> List[Field]:
        cls = type(self)
        return sorted(filter(lambda field: isinstance(field, field_cls), cls._instances.values()), key=lambda field: field.full_key)

    def get_all_indep(self) -> List[Indep]:
        return self._get_all_typed_fields(Indep)

    def get_all_dep(self) -> List[Dep]:
        return self._get_all_typed_fields(Dep)

    def get_all_header(self) -> List[Header]:
        return self._get_all_typed_fields(Header)

    def get_all_flag(self) -> List[Flag]:
        return self._get_all_typed_fields(Flag)


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

        if file_path.suffix != '.grid':
            raise ValueError(f'Can only deal with files with .grid suffix.')

        special_keywords = set()
        with file_path.open('r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip()
                is_flag = False
                # Try special keywords first.
                if line in ['matching_log_dir']:
                    special_keywords.add(line)
                    continue

                # Try nonflag pattern second.
                match = nonflag_pat.match(line)
                if match is None:
                    match = flag_pat.match(line)
                    is_flag = True
                if match is None:
                    raise FormatError(f"'{line}' doesn't match flag or nonflag patterns.")

                full_key = match.group(1)
                short_key = match.group(2)
                type_ = match.group(3)
                if not is_flag:
                    values = convert_values(match.group(4).split(','))
                    if type_ == 'dep' and len(values) > 1:
                        raise FormatError(f'Expect only one expression for dependent values.')
                    ff.create_field(type_, full_key, short_key, values=values)
                else:
                    ff.create_field(type_, full_key, short_key)

        if 'matching_log_dir' in special_keywords:
            value = f'log/grid/matched_cmdl/{file_path.stem}'
            if ff.count is None:
                ff.create_field('indep', 'log_dir', '', values=[value])
            else:
                ff.create_field('dep', 'log_dir', '', values=[value + f'/{{{ff.count.full_key}}}'])

        g = Graph()
        for f in ff.get_all_indep():
            g.add_vertice(f)
        for f in ff.get_all_dep():
            g.add_vertice(f)
            for dep in f.dep_fields:
                g.add_edge(dep, f)
        self.stack = g.topologicalSort()
        self.headers = ff.get_all_header()
        self.flags = ff.get_all_flag()

    def generate(self):

        value_dict = dict()
        full_keys = set(field.full_key for field in self.stack)
        key2cls = {field.full_key: type(field) for field in self.stack}
        header = ' '.join([h.value for h in self.headers]) + ' ' * bool(self.headers)
        flag = ' '.join([f'--{f.full_key}' for f in self.flags]) + ' ' * bool(self.flags)
        ret = list()

        def update(field, v):
            value_dict[field.full_key] = v
            if field.short_key:
                value_dict[field.short_key] = v

        def helper(ind: int):
            if ind >= len(self.stack):
                new_arg_lst = list()
                for k, v in sorted(value_dict.items()):
                    if k in full_keys:
                        cls = key2cls[k]
                        if cls is Indep or cls is Dep:
                            new_arg_lst.append(f'--{k} {v}')
                        elif cls is Flag:
                            new_arg_lst.append(f'--{k}')
                ret.append(header + flag + ' '.join(new_arg_lst))
                return

            field = self.stack[ind]
            if isinstance(field, Indep):  # This includes `Count`.
                for v in field.values:
                    update(field, v)
                    helper(ind + 1)
            elif isinstance(field, Flag):
                update(field, None)  # Use `None` to indicate it's a flag.
            else:
                v = eval(f"f'{field.value}'", {}, value_dict)
                update(field, v)
                helper(ind + 1)

        helper(0)
        return ret


if __name__ == "__main__":
    file_path = Path(sys.argv[1])
    grid = Grid(file_path)
    for setting in grid.generate():
        print(setting)
