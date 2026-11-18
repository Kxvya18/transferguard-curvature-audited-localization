"""Choosing which person to test next, using the mean-field likelihood.

After an outbreak is detected, a few observers have reported. Each further test
(an interview or a serology test) reveals one more person's infection time, or
that they were still healthy. Which person tells us the most about the source?

For every untested person v we compute the mutual information between the
source S and v's report Y_v:

    I(S; Y_v) = H(Y_v) - sum_s pi(s) H(Y_v | S = s),

where pi is the current posterior over sources and P(Y_v | S = s) comes from the
same mean-field propagation the localizer uses, averaged over start times and
transmission probabilities with their posterior weights. A test is worth the
most when the candidate sources disagree about what that person will report.
Belief propagation or simulation would need this for every person at every
step; mean field gets it for all of them in one pass.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .estimators import Context
from .mean_field import _contact_matrices, _start_times, mean_field_loglik
from .spread import earliest_arrival

PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=axis)


def _reports(ctx: Context, temperature: float = 2.0, max_sources: int = 100,
             prob_grid: tuple[float, ...] = PROBS, n_start_times: int = 8):
    """(mass, report): posterior mass of the most likely sources, and report[s, v, c], the
    mass-weighted probability that person v reports class c if s is the source. Class
    c < t_observe - t_lo is "infected at window t_lo + c"; the last class is "healthy"."""
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    loglik = mean_field_loglik(ctx, prob_grid=prob_grid, n_start_times=n_start_times) / temperature
    finite = np.isfinite(loglik)
    if not finite.any():
        return None
    weights = np.exp(np.where(finite, loglik - logsumexp(loglik[finite]), -np.inf))  # joint (start, prob, source)
    source_mass = weights.sum(axis=0)
    top = np.argsort(-source_mass)[:max_sources]
    top = top[source_mass[top] > 0]
    starts = _start_times(ctx, n_start_times)
    contacts = _contact_matrices(ctx)
    n_times = t_obs - ctx.t_lo
    report = np.zeros((len(top), n, n_times + 1))
    rows = np.arange(len(top))
    for i, tau in enumerate(starts):
        begin = np.maximum(tau, graph.join[top])
        for j, p in enumerate(prob_grid):
            w = weights[i * len(prob_grid) + j, top]
            if not w.any():
                continue
            q = np.zeros((len(top), n), dtype=np.float32)
            previous = np.zeros_like(q)
            for t in range(int(tau), t_obs):
                live = begin == t
                q[rows[live], top[live]] = 1.0
                report[:, :, t - ctx.t_lo] += w[:, None] * (q - previous)
                previous = q.copy()
                matrix = contacts[t - ctx.t_lo]
                if matrix is not None:
                    q = 1.0 - (1.0 - q) * np.exp(np.log1p(-np.float32(p) * q) @ matrix)
            report[:, :, -1] += w[:, None] * (1.0 - previous)  # not infected before t_observe
    return report[:, 0, :].sum(axis=1), report  # mass is the same whichever v is summed


def information_gain(ctx: Context, **kwargs) -> np.ndarray:
    """Mutual information between the source and each person's report (0 for people already tested)."""
    ob = ctx.outbreak
    found = _reports(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report = found
    conditional = report / np.maximum(mass, 1e-300)[:, None, None]
    marginal = report.sum(axis=0) / mass.sum()
    gain = _entropy(marginal) - (mass @ _entropy(conditional)) / mass.sum()
    gain[ob.observers] = 0.0
    return np.maximum(gain, 0.0)


def infection_probability(ctx: Context, **kwargs) -> np.ndarray:
    """Posterior probability that each person was infected before t_observe (0 for people
    already tested). Testing the most likely infected follows Baker et al. (PNAS 2021),
    who rank people for testing by their inferred infection probability."""
    found = _reports(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report = found
    prob = 1.0 - report[:, :, -1].sum(axis=0) / mass.sum()
    prob[ctx.outbreak.observers] = 0.0
    return prob


def candidate_split(ctx: Context) -> np.ndarray:
    """Spinelli et al. (2017)-style choice: the person whose earliest possible infection
    time differs most across the remaining candidate sources (entropy of that time over
    candidates, each weighted equally), so the report splits the candidates best."""
    ob, graph = ctx.outbreak, ctx.graph
    y = ob.observed_times
    infected = y < ob.t_observe
    arrival = np.minimum(earliest_arrival(graph, ctx.t_lo, ob.t_observe), ob.t_observe)
    candidates = (graph.join <= y[infected].min()) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1)
    sub = arrival[candidates]
    score = np.zeros(graph.n_nodes)
    for v in range(graph.n_nodes):
        _, counts = np.unique(sub[:, v], return_counts=True)
        score[v] = _entropy(counts / counts.sum())
    score[ob.observers] = -1.0
    return score
