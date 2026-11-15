"""Mean-field likelihood localizer for sparse, timed observers on an evolving network.

For every candidate source s at once, propagate

    q[s, v](t + 1) = 1 - (1 - q[s, v](t)) * prod over contacts u~v in window t of (1 - p * q[s, u](t)),

the mean-field probability that v is infected by time t if s started the outbreak.
An observer infected at time y contributes q(y) - q(y - 1) to the likelihood; an
observer still healthy at the observation time T contributes 1 - q(T). The
unknown start time and transmission probability are summed out over a grid, so
the method needs neither, and it needs no training and no random simulations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from .estimators import INFEASIBLE, Context
from .spread import earliest_arrival


def _contact_matrices(ctx: Context) -> list[sp.csr_matrix | None]:
    """Symmetric 0/1 contact matrix for each window in [t_lo, t_observe)."""
    n = ctx.graph.n_nodes
    out = []
    for t in range(ctx.t_lo, ctx.outbreak.t_observe):
        c = ctx.graph.windows[t]
        if not len(c):
            out.append(None)
            continue
        rows, cols = np.r_[c[:, 0], c[:, 1]], np.r_[c[:, 1], c[:, 0]]
        matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        matrix.data[:] = 1.0  # repeated contacts in one window count once
        out.append(matrix.astype(np.float32))
    return out


def mean_field_loglik(
    ctx: Context,
    prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    n_start_times: int = 8,
    floor: float = 1e-4,
) -> np.ndarray:
    """Log-likelihood of the observations for every node, shape (start times x probs, n_nodes)."""
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    y = ob.observed_times.astype(int)
    infected = y < t_obs
    first = int(y[infected].min())
    starts = np.unique(np.linspace(ctx.t_lo, max(first, ctx.t_lo), n_start_times).round().astype(int))
    contacts = _contact_matrices(ctx)
    # Only nodes that could have reached every infected observer in time can score above
    # the floor, so only those rows are propagated. Infection cannot beat the earliest
    # arrival, whatever the transmission probability, so this drops no possible source.
    arrival = earliest_arrival(graph, ctx.t_lo, t_obs)
    possible = np.flatnonzero((graph.join <= first) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1))
    rows = np.arange(len(possible))
    out = []
    for tau in starts:
        begin = np.maximum(tau, graph.join[possible])  # a node cannot start before it joins
        for p in prob_grid:
            q = np.zeros((len(possible), n), dtype=np.float32)
            at = np.zeros((len(possible), len(y)), dtype=np.float32)  # q at each observer's infection time
            before = np.zeros_like(at)  # q one step earlier
            for t in range(int(tau), t_obs + 1):
                live = begin == t
                q[rows[live], possible[live]] = 1.0
                current = q[:, ob.observers]
                at = np.where(y[None, :] == t, current, at)
                before = np.where(y[None, :] - 1 == t, current, before)
                if t == t_obs:
                    break
                matrix = contacts[t - ctx.t_lo]
                if matrix is not None:
                    q = 1.0 - (1.0 - q) * np.exp(np.log1p(-np.float32(p) * q) @ matrix)
            likelihood = np.where(infected[None, :], at - before, 1.0 - current)
            loglik = np.full(n, -np.inf)
            loglik[possible] = np.log(np.maximum(likelihood, floor)).sum(axis=1)
            loglik[possible[begin > first]] = -np.inf
            out.append(loglik)
    return np.array(out)


def mean_field_scores(ctx: Context, **kwargs) -> np.ndarray:
    """One score per node: log-likelihood with start time and probability summed out."""
    score = logsumexp(mean_field_loglik(ctx, **kwargs), axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)

