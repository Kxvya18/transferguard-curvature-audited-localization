"""Unbiased estimate of the exact likelihood P(reports | source, start, p) by a guided particle filter.

Plain Monte Carlo counts the runs whose every observer matches its report exactly; with many
observers almost no run matches, and the estimate is zero (High school: 28% of outbreaks).
Here every run is made to match. Window by window, each observer is forced to do what it
reported: stay healthy before its report time, be infected exactly at it (an observer reported
healthy is never infected before the observation time). The run is weighted by the probability
of the forced events; everyone else spreads as in the model. The weights' average is an
unbiased estimate of the likelihood (it is the likelihood of a particle filter whose proposal
is the model with the observers conditioned). Runs are resampled when their weights become
uneven (effective sample size below half), which keeps the estimate unbiased.

Same SI semantics as `spread.spread`: each listed contact is an independent chance; a person
infected at window t can pass it on in window t, and the contact is then infected at t + 1.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp


def contact_counts(graph, t_begin, t_end):
    """For each window, a sparse matrix C[u, v] = number of listed contacts u -> v (both directions)."""
    n = graph.n_nodes
    out = {}
    for t in range(t_begin, t_end):
        c = graph.windows[t]
        if len(c):
            rows, cols = np.r_[c[:, 0], c[:, 1]], np.r_[c[:, 1], c[:, 0]]
            out[t] = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return out


def log_likelihood(graph, outbreak, source, start, p, particles, rng, counts=None):
    """log of the estimated P(reports | source, start, p); -inf if impossible. Returns (value, final ESS share)."""
    n, t_obs = graph.n_nodes, outbreak.t_observe
    observers, y = outbreak.observers, outbreak.observed_times.astype(int)
    begin = max(int(start), int(graph.join[source]))
    if begin >= t_obs:
        return -np.inf, 0.0
    own = np.flatnonzero(observers == source)
    if len(own) and y[own[0]] != begin:  # the source is an observer: its report must be its start
        return -np.inf, 0.0
    if np.any((y <= begin) & (observers != source)):  # someone else reported infected before the start
        return -np.inf, 0.0
    counts = counts if counts is not None else contact_counts(graph, begin, t_obs)
    is_observer = np.zeros(n, dtype=bool)
    is_observer[observers] = True
    must_infect_at = np.full(n, -1)  # infection time forced for observers reported infected
    must_infect_at[observers[y < t_obs]] = y[y < t_obs]
    infected = np.zeros((particles, n), dtype=bool)
    infected[:, source] = True
    log_w = np.zeros(particles)
    log_z = 0.0
    log1mp = np.log1p(-p)
    for t in range(begin, t_obs - 1):  # infections at t + 1 <= t_obs - 1 matter; later ones look healthy
        c = counts.get(t)
        if c is not None:
            k = np.asarray((sp.csr_matrix(infected.astype(np.float64)) @ c).todense())  # infectious contacts per person
        else:
            k = np.zeros((particles, n))
        susceptible = ~infected
        escape_log = k * log1mp  # log P(escape this window)
        # everyone who is not an observer: spread as in the model
        catch = susceptible & ~is_observer[None, :] & (rng.random((particles, n)) >= np.exp(escape_log))
        # observers: forced
        forced_now = (must_infect_at == t + 1)[None, :] & susceptible
        stay = susceptible & is_observer[None, :] & ~forced_now
        with np.errstate(divide="ignore"):
            log_w += np.where(forced_now, np.log(-np.expm1(escape_log)), 0.0).sum(axis=1)
        log_w += np.where(stay, escape_log, 0.0).sum(axis=1)
        infected |= catch | forced_now
        alive = np.isfinite(log_w)
        if not alive.any():
            return -np.inf, 0.0
        w = np.exp(log_w - log_w[alive].max())
        ess = w.sum() ** 2 / (w ** 2).sum()
        if ess < particles / 2:
            log_z += logsumexp(log_w) - np.log(particles)
            pick = rng.choice(particles, size=particles, p=w / w.sum())
            infected = infected[pick]
            log_w = np.zeros(particles)
    alive = np.isfinite(log_w)
    if not alive.any():
        return -np.inf, 0.0
    w = np.exp(log_w - log_w[alive].max())
    return log_z + logsumexp(log_w) - np.log(particles), float(w.sum() ** 2 / (w ** 2).sum() / particles)
