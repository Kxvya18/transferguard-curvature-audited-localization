"""SI spreading over time windows and sparse observation of the outcome.

Semantics shared by every function here: a node infected at window t can pass
the infection along a contact in any window t' >= t, and the contacted node is
then infected at t' + 1. So infection moves at most one hop per window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .temporal_graph import TemporalGraph


def spread(
    graph: TemporalGraph,
    start: np.ndarray,
    t_begin: int,
    t_end: int,
    p: float | np.ndarray,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Run many SI outbreaks at once.

    `start` has shape (runs, n_nodes) and holds each node's infection time
    (np.inf if not yet infected). `p` is the per-contact transmission
    probability, either one value or one per run. Returns infection times after
    window t_end - 1. With `rng=None` every contact transmits (p is ignored),
    which gives the earliest possible arrival times.
    """
    times = start.copy()
    p_col = np.reshape(p, (-1, 1)) if np.ndim(p) else p
    for t in range(t_begin, t_end):
        contacts = graph.windows[t]
        if not len(contacts):
            continue
        for u, v in ((contacts[:, 0], contacts[:, 1]), (contacts[:, 1], contacts[:, 0])):
            can = (times[:, u] <= t) & (times[:, v] > t + 1)
            if rng is not None:
                can &= rng.random(can.shape) < p_col
            rows, cols = np.nonzero(can)
            np.minimum.at(times, (rows, v[cols]), t + 1)
    return times


def earliest_arrival(graph: TemporalGraph, t_begin: int, t_end: int) -> np.ndarray:
    """Earliest arrival time at every node from every possible source.

    Source s starts at max(t_begin, join[s]); row s of the result is its arrival map.
    """
    n = graph.n_nodes
    start = np.full((n, n), np.inf)
    start[np.arange(n), np.arange(n)] = np.maximum(t_begin, graph.join)
    return spread(graph, start, t_begin, t_end, p=1.0, rng=None)


@dataclass
class Outbreak:
    source: int
    t_start: int
    t_observe: int  # observations are taken at the end of window t_observe - 1
    observers: np.ndarray
    observed_times: np.ndarray  # infection time, or t_observe if not infected yet

    @property
    def infected_observers(self) -> np.ndarray:
        return self.observers[self.observed_times < self.t_observe]


def simulate_outbreak(
    graph: TemporalGraph,
    rng: np.random.Generator,
    p: float,
    horizon: int,
    observer_fraction: float,
    min_infected_observers: int = 3,
    max_tries: int = 200,
) -> Outbreak | None:
    """Draw a source active at a random window, spread for `horizon` windows, observe sparsely."""
    n = graph.n_nodes
    n_obs = max(int(round(observer_fraction * n)), min_infected_observers)
    for _ in range(max_tries):
        t_start = int(rng.integers(0, graph.n_windows - horizon))
        active = np.unique(graph.windows[t_start].ravel())
        if not len(active):
            continue
        source = int(rng.choice(active))
        start = np.full((1, n), np.inf)
        start[0, source] = t_start
        t_observe = t_start + horizon
        times = spread(graph, start, t_start, t_observe, p, rng)[0]
        observers = rng.choice(n, size=n_obs, replace=False)
        observed = np.minimum(times[observers], t_observe)
        if np.sum(observed < t_observe) >= min_infected_observers:
            return Outbreak(source, t_start, t_observe, observers, observed)
    return None
