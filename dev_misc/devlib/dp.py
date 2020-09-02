from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, wraps
from itertools import tee
from typing import Dict, Set, Tuple, TypeVar, Union

import torch
from networkx import DiGraph
from networkx.algorithms.dag import topological_sort

from .tensor_x import Renamed
from .tensor_x import TensorX as Tx

# A decorator for creating a proper state data class.
make_state = partial(dataclass, frozen=True)


class StateUnreachable(Exception):
    """Raise this if there is some unreachable state."""


State = TypeVar('State')
_StateLike = TypeVar('StateLike')


def use_grad_switch(func):

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if self.grad_enabled:
            with torch.enable_grad():
                return func(self, *args, **kwargs)
        else:
            with torch.no_grad():
                return func(self, *args, **kwargs)

    return wrapped


class BaseDP(ABC):

    def __init__(self, grad_enabled: bool = False):
        self._G = DiGraph()
        self._base_states: Set[State] = set()
        self._decisions: Dict[State, Set[Tuple[State]]] = defaultdict(set)
        self._grad_enabled = grad_enabled

    # NOTE(j_luo) For some reason, tensors stored in `G` are not collected and freed (probably due to circular reference).
    def __del__(self):
        self._G.clear()

    @property
    def grad_enabled(self) -> bool:
        return self._grad_enabled

    def add_node(self, state: State):
        self._G.add_node(state, value=None)

    def add_decision(self, out_state: State, *in_states: State):
        if not in_states:
            raise TypeError(f'Must have at least one in state.')

        for in_state in in_states:
            self._G.add_edge(in_state, out_state)

        self._decisions[out_state].add(in_states)

    @classmethod
    def _get_state(cls, key: _StateLike) -> State:
        return key

    def __getitem__(self, key: _StateLike) -> Tx:
        cls = type(self)
        key = cls._get_state(key)
        return self._G.nodes[key]['value']

    def __setitem__(self, key: _StateLike, value: Tx):
        cls = type(self)
        key = cls._get_state(key)
        self._G.nodes[key]['value'] = value

    @use_grad_switch
    def run(self):
        # Run topo sort first.
        o1, o2 = tee(topological_sort(self._G))

        # Check if all end states are reached.
        reachable = set()
        reachable.update(self._base_states)
        for state in o1:
            if state not in reachable:
                raise StateUnreachable(f'{state} unreachable.')
            for _, out_state in self._G.edges(state):
                reachable.add(out_state)

        # Main loop.
        for state in o2:
            if state in self._base_states:
                continue

            self.update_state(state)

    @abstractmethod
    def update_state(self, state: State):
        """Update state value."""


@make_state
class FibState:
    i: int


class Fibonacci(BaseDP):

    def __init__(self, a0: Tx, a1: Tx, size: int, grad_enabled: bool = False):
        super().__init__(grad_enabled)

        for i in range(size + 1):
            self.add_node(FibState(i))

        for i in range(2, size + 1):
            self.add_decision(FibState(i), FibState(i - 1), FibState(i - 2))

        # FIXME(j_luo) add check that it's added.
        self._base_states = {FibState(0), FibState(1)}
        self[0] = a0
        self[1] = a1

    def update_state(self, state: FibState):
        s = 0
        for in_state, _ in self._G.in_edges(state):
            s += self[in_state]
        self[state] = s

    @classmethod
    def _get_state(cls, key: Union[int, FibState]) -> FibState:
        if isinstance(key, int):
            key = FibState(key)
        return key


@make_state
class HmmPMState:
    """Posterior marginal for HMM."""
    t: int


@make_state
class HmmFState:
    """Forward state for HMM."""
    t: int


@make_state
class HmmBState:
    """Backward state for HMM."""
    t: int


HmmState = TypeVar('HmmState', HmmPMState, HmmFState, HmmBState)


class Hmm(BaseDP):
    """Inference for Hmm using Forward-Backword algorithm:

    Pr( y[t] | x[1:T] ) =
    Pr( y[t] | x[1:t] ) * Pr( x[t+1:T] | y[t] ) / Z
    = Forward( t, y[t] ) * Backward( t, y[t] ) / Z

    Forward( t, y[t] ) =
    Emission( y[t], x[t] ) * Sum_y[t-1]{ Forward( t-1, y[t-1] ) * Transition( y[t-1], y[t] ) } / Z
    Base case:
    Forward( 1, y[1] ) =
    Emission( y[1], x[1] ) * p0 / Z


    Backward( t, y[t] ) =
    Sum_y[t+1]{ Transition( y[t], y[t+1] ) * Emission( y[t+1], x[t+1] ) * Backward( t+1, y[t+1] ) }
    """

    def __init__(self, obs: Tx, p0: Tx, tp: Tx, ep: Tx, grad_enabled: bool = False):
        super().__init__(grad_enabled)

        self._obs = obs
        bs = obs.size('batch')
        l = obs.size('l')

        for t in range(l):
            self.add_node(HmmPMState(t))
            self.add_node(HmmFState(t))
            self.add_node(HmmBState(t))

        for t in range(l):
            self.add_decision(HmmPMState(t), HmmFState(t), HmmBState(t))
            # FIXME(j_luo) Add boundary check?
            if t > 0:
                self.add_decision(HmmFState(t), HmmFState(t - 1))
            if t < l - 1:
                self.add_decision(HmmBState(t), HmmBState(t + 1))

        self._base_states = {HmmFState(0), HmmBState(l - 1)}

        first_obs = obs.select('l', 0)  # dim: batch
        pr_x0_g_y0 = ep.batched_select('x', first_obs)  # dim: y, batch
        self[HmmFState(0)] = (pr_x0_g_y0 * p0).normalize_prob('y')  # dim: y, batch
        self[HmmBState(l - 1)] = p0.ones_like()

        self._tp = tp
        self._ep = ep

    @classmethod
    def _get_state(cls, key: Union[Tuple[str, int], HmmState]) -> HmmState:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError(f'Tuple key must be of length 2.')
            name, t = key
            if name == 'pm':
                state_cls = HmmPMState
            elif name == 'f':
                state_cls = HmmFState
            elif name == 'b':
                state_cls = HmmBState
            else:
                raise ValueError(f'Unrecognized value "{name}" for name.')
            key = state_cls(t)
        return key

    def update_state(self, state: HmmState):
        t = state.t
        if isinstance(state, HmmPMState):
            v = (self['f', t] * self['b', t]).normalize_prob('y')
            self['pm', t] = v
        elif isinstance(state, HmmFState):
            curr_obs = self._obs.select('l', t)
            pr_xt_g_y = self._ep.batched_select('x', curr_obs)
            f = self['f', t - 1]
            v = (f.contract(self._tp, 'y').rename_({'yn': 'y'}) * pr_xt_g_y).normalize_prob('y')
            self['f', t] = v
        elif isinstance(state, HmmBState):
            last_obs = self._obs.select('l', t + 1)
            pr_xtp1_g_y = self._ep.batched_select('x', last_obs)
            b = self['b', t + 1]
            take_yn = {'y': 'yn'}
            with Renamed(pr_xtp1_g_y, take_yn), Renamed(b, take_yn):
                v = self._tp.contract(pr_xtp1_g_y * b, 'yn').normalize_prob('y')
            self['b', t] = v
        else:
            raise TypeError(f'Unrecognized type "{type(state)}" for state.')


@make_state
class LisState:
    i: int


class Lis(BaseDP):
    """Longest increasing subsequence."""

    def __init__(self, a: Tx, grad_enabled: bool = False):
        super().__init__(grad_enabled)

        self._a = a
        l = a.size('l')

        for i in range(l + 1):
            self.add_node(LisState(i))

        for i in range(l + 1):
            for j in range(i):
                self.add_decision(LisState(i), LisState(j))

        self._base_states = {LisState(0)}
        a1 = a.select('l', 0)
        self[0] = a1.zeros_like()
        self._a0 = a1.full_like(a.max() - 1)

    @classmethod
    def _get_state(cls, key: Union[int, LisState]) -> LisState:
        if isinstance(key, int):
            key = LisState(key)
        return key

    def update_state(self, state: LisState):
        decisions = list()
        if state.i == 0:
            ai = self._a0
        else:
            ai = self._a.select('l', state.i - 1)
        for in_state, _ in self._G.in_edges(state):
            al = self._a.select('l', in_state.i - 1)
            v = self[in_state.i] + (al < ai).float()
            decisions.append(v)
        self[state] = Tx.max_of(decisions)[0]


@make_state
class CmmState:
    left: int
    right: int


class Cmm(BaseDP):
    """Chain matrix multiplication."""

    def __init__(self, lengths: Tx, widths: Tx, grad_enabled: bool = False):
        super().__init__(grad_enabled)

        self._lengths = lengths
        self._widths = widths
        bs = self._lengths.size('batch')
        l = self._lengths.size('l')
        for i in range(l):
            for j in range(i, l):
                self.add_node(CmmState(i, j))

        for i in range(l):
            for j in range(i, l):
                for k in range(i, j):
                    self.add_decision(CmmState(i, j), CmmState(i, k), CmmState(k + 1, j))

        self._base_states = set()
        zeros = self._lengths.select('l', 0).zeros_like()
        for i in range(l):
            single = CmmState(i, i)
            self._base_states.add(single)
            self[single] = zeros

    @classmethod
    def _get_state(cls, key: Union[Tuple[int, int], CmmState]) -> CmmState:
        if isinstance(key, tuple):
            l, r = key
            key = CmmState(l, r)
        return key

    def update_state(self, state: CmmState):
        decisions = list()
        i, j = state.left, state.right
        # FIXME(j_luo) Use decisions for all.
        for in_states in self._decisions[state]:
            l_state, r_state = in_states
            m = self._lengths.select('l', l_state.left)
            n = self._widths.select('l', l_state.right)
            p = self._widths.select('l', r_state.right)
            decisions.append(self[l_state] + self[r_state] + m * n * p)
        self[state] = Tx.min_of(decisions)[0]


@make_state
class EditDistState:  # FIXME(j_luo) Automatic state register?
    first: int
    second: int


class EditDist(BaseDP):

    def __init__(self, string0: Tx, string1: Tx, length0: Tx, length1: Tx, penalty: Tx = None, grad_enabled: bool = False):
        super().__init__(grad_enabled)

        bs = string0.size('batch')
        l0 = string0.size("l")
        l1 = string1.size('l')

        self._string0 = string0
        self._string1 = string1
        self._length0 = length0
        self._length1 = length1
        self._penalty = penalty

        for i in range(l0 + 1):
            for j in range(l1 + 1):
                self.add_node(EditDistState(i, j))

        for i in range(l0 + 1):
            for j in range(l1 + 1):
                if i > 0:
                    # FIXME(j_luo) Decision should have types/kinds.
                    self.add_decision(EditDistState(i, j), EditDistState(i - 1, j))
                if j > 0:
                    self.add_decision(EditDistState(i, j), EditDistState(i, j - 1))
                if i > 0 and j > 0:
                    self.add_decision(EditDistState(i, j), EditDistState(i - 1, j - 1))

        self._base_states = {EditDistState(0, 0)}
        self[0, 0] = self._length0.zeros_like()

    @classmethod
    def _get_state(cls, key: Union[Tuple[int, int], EditDistState]):
        if isinstance(key, tuple):
            first, second = key
            return EditDistState(first, second)
        return key

    def update_state(self, state: EditDistState):
        decisions = list()
        i, j = state.first, state.second
        for in_states in self._decisions[state]:
            in_state = in_states[0]
            if in_state.first == i or in_state.second == j:
                decisions.append(self[in_state] + 1)
            else:
                s0 = self._string0.select('l', i - 1)
                s1 = self._string1.select('l', j - 1)
                if self._penalty is not None:
                    penalty = self._penalty.rename(None)[s0.data.rename(None), s1.data.rename(None)]
                    penalty = Tx(penalty, ['batch'])
                else:
                    penalty = (s0 != s1).float()
                decisions.append(self[in_state] + penalty)
        self[state] = Tx.min_of(decisions)[0]

    @use_grad_switch
    def make_grid(self) -> Tx:
        l0 = self._string0.size('l')
        l1 = self._string1.size('l')
        grid = list()
        for i in range(l0 + 1):
            for j in range(l1 + 1):
                grid.append(self[i, j])
        grid = Tx.stack(grid, 'grid')
        grid = grid.unflatten('grid', [('src_l', l0 + 1), ('tgt_l', l1 + 1)])
        return grid

    @use_grad_switch
    def get_results(self) -> Tx:
        grid = self.make_grid()
        ret = grid.each_select({'src_l': self._length0, 'tgt_l': self._length1})
        return ret
