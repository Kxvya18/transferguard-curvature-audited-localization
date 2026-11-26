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
from .spread import earliest_arrival, latest_departure


def _contact_matrices(ctx: Context, directed: bool = False, weighted: bool = False) -> list[sp.csr_matrix | None]:
    """0/1 contact matrix for each window in [t_lo, t_observe): symmetric, or with
    `directed`, row u -> column v only (pairs are then read as u passes it to v). With
    `weighted`, a pair listed k times in a window counts as k independent contacts."""
    n = ctx.graph.n_nodes
    out = []
    for t in range(ctx.t_lo, ctx.outbreak.t_observe):
        c = ctx.graph.windows[t]
        if not len(c):
            out.append(None)
            continue
        rows, cols = (c[:, 0], c[:, 1]) if directed else (np.r_[c[:, 0], c[:, 1]], np.r_[c[:, 1], c[:, 0]])
        matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        if not weighted:
            matrix.data[:] = 1.0  # repeated contacts in one window count once
        out.append(matrix.astype(np.float32))
    return out


def _start_times(ctx: Context, n_start_times: int | None) -> np.ndarray:
    """Candidate start windows between t_lo and the first observed infection (all of them if None)."""
    first = int(ctx.infected_times.min())
    if n_start_times is None:
        return np.arange(ctx.t_lo, max(first, ctx.t_lo) + 1)
    return np.unique(np.linspace(ctx.t_lo, max(first, ctx.t_lo), n_start_times).round().astype(int))


def mean_field_marginals(graph, source: int, t_begin: int, t_end: int, p: float) -> np.ndarray:
    """Mean-field infection probabilities q[t, v] for t = t_begin .. t_end, one known source.

    These upper-bound the exact SI probabilities (Lemma 1 in the paper): whether a node is
    infected is an increasing function of the independent per-contact transmission coins,
    so by Harris's inequality P(v healthy at t + 1) >= P(v healthy at t) * prod (1 - p P(u infected at t)),
    and induction over windows gives exact <= q.
    """
    q = np.zeros(graph.n_nodes)
    q[source] = 1.0
    out = [q.copy()]
    for t in range(t_begin, t_end):
        c = graph.windows[t]
        if len(c):
            pairs = np.unique(np.sort(c, axis=1), axis=0)
            log_survive = np.zeros(graph.n_nodes)
            np.add.at(log_survive, pairs[:, 1], np.log1p(-p * q[pairs[:, 0]]))
            np.add.at(log_survive, pairs[:, 0], np.log1p(-p * q[pairs[:, 1]]))
            q = 1.0 - (1.0 - q) * np.exp(log_survive)
        out.append(q.copy())
    return np.array(out)


def mean_field_loglik(
    ctx: Context,
    prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    n_start_times: int | None = 8,
    floor: float = 1e-4,
    status_only: bool = False,
    prune: bool = True,
    recovery_grid: tuple[float, ...] = (0.0,),
    susceptible_grid: tuple[float, ...] = (1.0,),
    seed_prob: float | None = None,
    directed: bool = False,
    candidates: np.ndarray | None = None,
    leak: float = 0.0,
    weighted: bool = False,
    detection: np.ndarray | None = None,
    false_report: np.ndarray | None = None,
    report_width: int = 1,
    detectability: np.ndarray | None = None,
    batch: int | None = None,
) -> np.ndarray:
    """Log-likelihood of the observations for every node, shape (start times x probs, n_nodes),
    start-time major (row = start index * len(prob_grid) + prob index).

    With `prune=False`, no candidate is ruled out for failing to reach an infected
    observer in time: that report then just scores the floor, as if it came from
    outside the network. Use this on real data, where some infections do.

    `recovery_grid` lets an infected node stop spreading: each window it stays
    infectious with probability 1 - r (r = 0, the default, is SI; r = 1 gives each
    new case one window to pass it on). `susceptible_grid` lets only a fraction sigma of
    people catch the infection at all (the rest are immune or inactive); sigma = 1, the
    default, is everyone. Rows run over start x prob x recovery x susceptible fraction.

    `seed_prob` gives the source its own transmission probability (a broadcaster can
    reach far more people per contact than anyone else); None means the same as p.
    `directed` reads contact pairs as one-way. `candidates` restricts which sources are
    scored (e.g. only the true one, when fitting the model on past outbreaks).
    `leak` is the chance per window that a susceptible person is infected from outside
    the network; with it, every report has a proper probability and `floor` can be ~0.
    `weighted`: see `_contact_matrices`. `detection[j, t]` is the chance that observer j,
    once infected, is detected in window t (e.g. from how many tests or sequences it ran
    then); a report is then the first detection, not the infection itself.
    `false_report[j, t]` (with `detection`) is the chance observer j wrongly reports the
    infection in window t whether or not it is infected (e.g. a misassigned sequence).
    `report_width` (with `detection`): a report at y means "first detected in windows
    y .. y + width - 1", for reports coarser than the model's time step.
    `detectability[a]` (with `detection`) scales how detectable an observer is a windows
    after its infection (e.g. local prevalence growing): the detection chance becomes
    1 - (1 - detection) ** detectability[a]; ages past the end use the last value.

    With `status_only`, a test says only "infected" or "healthy" at t_observe, not when:
    infected observers must then be given time t_observe - 1, and contribute q(t_observe - 1).

    `batch`: candidates are propagated `batch` at a time, so memory is about batch x n per
    array instead of candidates x n (results are identical). None: 256, fewer on very large
    networks so that one array stays near 80 MB."""
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    y = ob.observed_times.astype(int)
    infected = y < t_obs
    first = int(y[infected].min())
    starts = _start_times(ctx, n_start_times)
    contacts = _contact_matrices(ctx, directed, weighted)
    # Only nodes that could have reached every infected observer in time can score above
    # the floor, so only those rows are propagated. Infection cannot beat the earliest
    # arrival, whatever the transmission probability, so this drops no possible source.
    if prune:
        # backward from each infected observer: size observers x n, not n x n
        latest = latest_departure(graph, ctx.t_lo, ob.observers[infected], y[infected])
        begin_any = np.maximum(ctx.t_lo, graph.join)
        possible = np.flatnonzero((graph.join <= first) & np.all(begin_any[None, :] <= latest, axis=0))
    else:
        possible = np.flatnonzero(graph.join <= first)
    if candidates is not None:
        possible = np.intersect1d(possible, candidates)
    if batch is None:
        batch = max(8, min(256, 20_000_000 // n))
    if len(possible) > batch:
        kwargs = dict(prob_grid=prob_grid, n_start_times=n_start_times, floor=floor, status_only=status_only,
                      prune=False, recovery_grid=recovery_grid, susceptible_grid=susceptible_grid,
                      seed_prob=seed_prob, directed=directed, leak=leak, weighted=weighted, detection=detection,
                      false_report=false_report, report_width=report_width, detectability=detectability, batch=batch)
        parts = [mean_field_loglik(ctx, candidates=possible[i:i + batch], **kwargs) for i in range(0, len(possible), batch)]
        out = parts[0]
        for part in parts[1:]:
            out = np.where(np.isneginf(out), part, out)
        return out
    rows = np.arange(len(possible))
    out = []
    for tau in starts:
        begin = np.maximum(tau, graph.join[possible])  # a node cannot start before it joins
        for p, r, sigma in [(p, r, g) for p in prob_grid for r in recovery_grid for g in susceptible_grid]:
            q = np.zeros((len(possible), n), dtype=np.float32)
            spreading = np.zeros_like(q)  # probability of being infected and still infectious
            seed_spreading = np.zeros(len(possible), dtype=np.float32)  # used only with seed_prob
            at = np.zeros((len(possible), len(y)), dtype=np.float32)  # q at each observer's infection time
            before = np.zeros_like(at)  # q one step earlier
            ages = 1 if detectability is None else len(detectability)
            undetected = np.zeros((len(possible), len(y), ages), dtype=np.float32)  # by age since infection
            previous = np.zeros_like(at)
            found = np.zeros_like(at)  # with `detection`: truly detected so far
            no_false = np.ones(len(y), dtype=np.float32)  # no false report so far
            silent = np.ones_like(at)  # nothing reported so far, after the previous window
            for t in range(int(tau), t_obs + 1):
                live = begin == t
                q[rows[live], possible[live]] = 1.0
                if seed_prob is None:
                    spreading[rows[live], possible[live]] = 1.0
                else:
                    seed_spreading[live] = 1.0
                current = q[:, ob.observers]
                if detection is not None:
                    # newly infected join the undetected pool, then each is detected with chance h;
                    # independently, a false report comes with chance f
                    if ages > 1:  # everyone undetected gets one window older; the last age keeps its own
                        undetected[:, :, -1] += undetected[:, :, -2]
                        undetected[:, :, 1:-1] = undetected[:, :, :-2].copy()
                        undetected[:, :, 0] = 0.0
                    undetected[:, :, 0] += current - previous
                    previous = current
                    k = min(t, detection.shape[1] - 1)
                    if ages > 1:
                        chance = 1.0 - (1.0 - detection[:, k, None]) ** detectability[None, :].astype(np.float32)
                    else:
                        chance = detection[:, k, None]
                    caught_by_age = undetected * chance[None, :, :]
                    undetected -= caught_by_age
                    caught = caught_by_age.sum(axis=2)
                    found += caught
                    if false_report is not None:
                        no_false = no_false * (1.0 - false_report[:, k])
                    now_silent = no_false[None, :] * (1.0 - found)
                    before = np.where(y[None, :] == t, silent, before)  # silent until window y
                    at = np.where(y[None, :] + report_width - 1 == t, now_silent, at)  # still silent after the report period
                    silent = now_silent
                    if t == t_obs - 1:
                        still_healthy = now_silent  # nothing reported before t_observe
                else:
                    at = np.where(y[None, :] == t, current, at)
                    before = np.where(y[None, :] - 1 == t, current, before)
                    if t == t_obs - 1:
                        still_healthy = 1.0 - current  # reported healthy = not infected before t_observe
                if t == t_obs:
                    break
                matrix = contacts[t - ctx.t_lo]
                new = 0.0
                if matrix is not None or leak > 0:
                    # a susceptible person not yet infected (probability sigma - q) escapes this window
                    escape = np.float32(1.0 - leak)
                    if matrix is not None:
                        log_escape = np.log1p(-np.float32(p) * spreading) @ matrix
                        if seed_prob is not None:
                            log_escape += matrix[possible].multiply(np.log1p(-np.float32(seed_prob) * seed_spreading)[:, None]).toarray()
                        escape = escape * np.exp(log_escape)
                    updated = np.maximum(q, np.float32(sigma) - (np.float32(sigma) - q) * escape)
                    new, q = updated - q, updated
                spreading = np.float32(1.0 - r) * spreading + new
                seed_spreading *= np.float32(1.0 - r)
            hit = before - at if detection is not None else (at if status_only else at - before)
            likelihood = np.where(infected[None, :], hit, still_healthy)
            loglik = np.full(n, -np.inf)
            loglik[possible] = np.log(np.maximum(likelihood, floor)).sum(axis=1)
            loglik[possible[begin > first]] = -np.inf
            out.append(loglik)
    return np.array(out)


def mean_field_scores(ctx: Context, **kwargs) -> np.ndarray:
    """One score per node: log-likelihood with start time and probability summed out."""
    score = logsumexp(mean_field_loglik(ctx, **kwargs), axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)


def contact_start_scores(ctx: Context, prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9), **kwargs) -> np.ndarray:
    """Mean-field scores under the prior that the outbreak starts at a contact event:
    the source s and start window tau are drawn uniformly from the (person, window)
    pairs in which that person had a contact. Every start window is evaluated."""
    loglik = mean_field_loglik(ctx, prob_grid=prob_grid, n_start_times=None, **kwargs)
    starts = _start_times(ctx, None)
    prior = np.full((len(starts), ctx.graph.n_nodes), -np.inf)
    for i, tau in enumerate(starts):
        active = np.unique(ctx.graph.windows[tau].ravel())
        if len(active):
            prior[i, active] = -np.log(len(active))
    score = logsumexp(loglik + np.repeat(prior, len(prob_grid), axis=0), axis=0)
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
    status_only: bool = False,
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
            likelihood = np.where(infected[None, :], at if status_only else at - before, still_healthy)
            loglik = np.full(n, -np.inf)
            loglik[possible] = np.log(np.maximum(likelihood, floor)).sum(axis=1)
            loglik[possible[begin > first]] = -np.inf
            out.append(loglik)
    return np.array(out)


def message_passing_scores(ctx: Context, **kwargs) -> np.ndarray:
    """One score per node from `message_passing_loglik`, start time and probability summed out."""
    score = logsumexp(message_passing_loglik(ctx, **kwargs), axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)


def filtered_loglik(
    ctx: Context,
    prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    n_start_times: int | None = 8,
    floor: float = 1e-4,
) -> np.ndarray:
    """Mean field conditioned on the observations as they arrive (forward filtering).

    `mean_field_loglik` multiplies each observer's own probability, as if the
    observers were independent. They are not: one infection chain links them. Here
    the likelihood is built window by window, P(y) = prod_t P(reports at t + 1 | reports
    up to t), and after each window every observer is set to what it reported:
    infected (so it spreads for certain) or healthy (so it blocks paths through it).
    Same output shape as `mean_field_loglik`.
    """
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    y = ob.observed_times.astype(int)
    infected = y < t_obs
    first = int(y[infected].min())
    starts = _start_times(ctx, n_start_times)
    contacts = _contact_matrices(ctx)
    arrival = earliest_arrival(graph, ctx.t_lo, t_obs)
    possible = np.flatnonzero((graph.join <= first) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1))
    rows = np.arange(len(possible))
    # A candidate that is itself an observer is scored by its own report only through
    # consistency: it must have reported exactly its start time.
    slot = np.full(n, -1)
    slot[ob.observers] = np.arange(len(y))
    own_slot = slot[possible]
    out = []
    for tau in starts:
        begin = np.maximum(tau, graph.join[possible])
        own_ok = (own_slot < 0) | (y[np.maximum(own_slot, 0)] == begin)
        own_mask = np.zeros((len(possible), len(y)), dtype=bool)
        own_mask[rows[own_slot >= 0], own_slot[own_slot >= 0]] = True
        for p in prob_grid:
            q = np.zeros((len(possible), n), dtype=np.float32)
            loglik = np.zeros(len(possible))
            for t in range(int(tau), t_obs):
                live = begin == t
                q[rows[live], possible[live]] = 1.0
                q[:, ob.observers] = (y <= t).astype(np.float32)[None, :]
                matrix = contacts[t - ctx.t_lo]
                if matrix is not None:
                    survive = np.exp(np.log1p(-np.float32(p) * q) @ matrix)
                else:
                    survive = np.ones_like(q)
                s_obs = survive[:, ob.observers]
                stays = (t + 1 < y)[None, :]  # must survive this window
                caught = ((t + 1 == y) & infected)[None, :]  # must be infected by this window
                term = np.where(stays, np.log(np.maximum(s_obs, floor)), 0.0)
                term = np.where(caught, np.log(np.maximum(1.0 - s_obs, floor)), term)
                term[own_mask] = 0.0
                loglik += term.sum(axis=1)
                q = 1.0 - (1.0 - q) * survive
            full = np.full(n, -np.inf)
            full[possible] = loglik
            full[possible[(begin > first) | ~own_ok]] = -np.inf
            out.append(full)
    return np.array(out)


def filtered_scores(ctx: Context, **kwargs) -> np.ndarray:
    """One score per node from `filtered_loglik`, start time and probability summed out."""
    score = logsumexp(filtered_loglik(ctx, **kwargs), axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)
