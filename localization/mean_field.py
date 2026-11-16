"""Mean-field likelihood localizer for sparse, timed observers on an evolving network.

For every candidate source s at once, propagate

    q[s, v](t + 1) = 1 - (1 - q[s, v](t)) * prod over contacts u~v in window t of (1 - p * q[s, u](t)),

the mean-field probability that v is infected by time t if s started the outbreak.
An observer infected at time y contributes q(y) - q(y - 1) to the likelihood; an
observer reported healthy at the observation time T (not infected before T)
contributes 1 - q(T - 1). The
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
                if t == t_obs - 1:
                    still_healthy = 1.0 - current  # reported healthy = not infected before t_observe
                if t == t_obs:
                    break
                matrix = contacts[t - ctx.t_lo]
                if matrix is not None:
                    q = 1.0 - (1.0 - q) * np.exp(np.log1p(-np.float32(p) * q) @ matrix)
            likelihood = np.where(infected[None, :], at - before, still_healthy)
            loglik = np.full(n, -np.inf)
            loglik[possible] = np.log(np.maximum(likelihood, floor)).sum(axis=1)
            loglik[possible[begin > first]] = -np.inf
            out.append(loglik)
    return np.array(out)


def mean_field_scores(ctx: Context, **kwargs) -> np.ndarray:
    """One score per node: log-likelihood with start time and probability summed out."""
    score = logsumexp(mean_field_loglik(ctx, **kwargs), axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)



def _directed_edges(ctx: Context):
    """Directed contact edges seen in [t_lo, t_observe), their reverses, and per-window activity."""
    n = ctx.graph.n_nodes
    windows = [ctx.graph.windows[t] for t in range(ctx.t_lo, ctx.outbreak.t_observe)]
    pairs = np.concatenate([w for w in windows if len(w)] or [np.zeros((0, 2), dtype=np.int64)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    src = np.r_[pairs[:, 0], pairs[:, 1]]
    dst = np.r_[pairs[:, 1], pairs[:, 0]]
    m = len(pairs)
    reverse = np.r_[np.arange(m, 2 * m), np.arange(m)]
    key = src * n + dst
    order = np.argsort(key)
    active = []
    for w in windows:
        if not len(w):
            active.append(None)
            continue
        k = np.r_[w[:, 0] * n + w[:, 1], w[:, 1] * n + w[:, 0]]
        active.append(np.unique(order[np.searchsorted(key[order], k)]))
    # incoming[e, i] = 1 if edge e points into node i: sums messages arriving at each node.
    incoming = sp.csr_matrix((np.ones(2 * m, dtype=np.float32), (np.arange(2 * m), dst)), shape=(2 * m, n))
    return src, dst, reverse, active, incoming


def message_passing_loglik(
    ctx: Context,
    prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    n_start_times: int = 8,
    floor: float = 1e-4,
) -> np.ndarray:
    """Like `mean_field_loglik`, but with dynamic message passing (Lokhov et al., 2014)
    run on the evolving contacts, for every candidate source at once.

    Mean field lets an infection bounce straight back to the node it came from, which
    inflates infection probabilities where the same pairs meet repeatedly. Message
    passing keeps, for each directed contact i -> j, the probability that i is infected
    without counting what it could have caught from j, which removes that echo.

    Per directed edge: theta = P(no infection has passed i -> j yet), phi = P(i is infected
    in the graph without j and has not yet passed it to j). In a window where the edge is
    active, theta -= p * phi and phi = (1 - p) * phi - (change in i's cavity susceptibility).
    """
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    y = ob.observed_times.astype(int)
    infected = y < t_obs
    first = int(y[infected].min())
    starts = np.unique(np.linspace(ctx.t_lo, max(first, ctx.t_lo), n_start_times).round().astype(int))
    arrival = earliest_arrival(graph, ctx.t_lo, t_obs)
    possible = np.flatnonzero((graph.join <= first) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1))
    rows = np.arange(len(possible))
    src, dst, reverse, active, incoming = _directed_edges(ctx)
    tiny = np.float32(1e-30)

    def cavity_susceptible(log_theta, seed_alive):
        """P_S^{i->j} for every edge: i susceptible, ignoring what it could catch from j."""
        log_in = log_theta @ incoming  # sum of log theta over edges into each node
        log_in = np.asarray(log_in)
        return seed_alive[:, src] * np.exp(log_in[:, src] - log_theta[:, reverse])

    out = []
    for tau in starts:
        begin = np.maximum(tau, graph.join[possible])
        for p in prob_grid:
            k = len(possible)
            theta = np.ones((k, len(src)), dtype=np.float32)
            phi = np.zeros_like(theta)
            seed_alive = np.ones((k, n), dtype=np.float32)  # 0 once the candidate itself has started
            ps_edge = np.ones_like(theta)
            at = np.zeros((k, len(y)), dtype=np.float32)
            before = np.zeros_like(at)
            for t in range(int(tau), t_obs + 1):
                live = begin == t
                if live.any():
                    seed_alive[rows[live], possible[live]] = 0.0
                    new_ps = cavity_susceptible(np.log(np.maximum(theta, tiny)), seed_alive)
                    phi += ps_edge - new_ps
                    ps_edge = new_ps
                log_theta = np.log(np.maximum(theta, tiny))
                infected_by_t = 1.0 - seed_alive * np.exp(np.asarray(log_theta @ incoming))
                current = infected_by_t[:, ob.observers]
                at = np.where(y[None, :] == t, current, at)
                before = np.where(y[None, :] - 1 == t, current, before)
                if t == t_obs - 1:
                    still_healthy = 1.0 - current  # reported healthy = not infected before t_observe
                if t == t_obs:
                    break
                edges = active[t - ctx.t_lo]
                if edges is None:
                    continue
                theta[:, edges] -= p * phi[:, edges]
                phi[:, edges] *= 1.0 - p
                new_ps = cavity_susceptible(np.log(np.maximum(theta, tiny)), seed_alive)
                phi -= new_ps - ps_edge
                ps_edge = new_ps
            likelihood = np.where(infected[None, :], at - before, still_healthy)
            loglik = np.full(n, -np.inf)
            loglik[possible] = np.log(np.maximum(likelihood, floor)).sum(axis=1)
            loglik[possible[begin > first]] = -np.inf
            out.append(loglik)
    return np.array(out)


def message_passing_scores(ctx: Context, **kwargs) -> np.ndarray:
    """One score per node from `message_passing_loglik`, start time and probability summed out."""
    score = logsumexp(message_passing_loglik(ctx, **kwargs), axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)
