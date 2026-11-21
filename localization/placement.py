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
from .spread import earliest_arrival, spread

PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=axis)


def _reports(ctx: Context, temperature: float = 2.0, max_sources: int = 100,
             prob_grid: tuple[float, ...] = PROBS, n_start_times: int = 8):
    """(mass, report, within): posterior mass of the most likely sources; report[s, v, c], the
    mass-weighted probability that person v reports class c if s is the source (class
    c < t_observe - t_lo is "infected at window t_lo + c", the last class is "healthy");
    and within[v], the mass-weighted entropy of v's report under each full hypothesis
    (source, start time, probability)."""
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
    within = np.zeros(n)
    rows = np.arange(len(top))
    for i, tau in enumerate(starts):
        begin = np.maximum(tau, graph.join[top])
        for j, p in enumerate(prob_grid):
            w = weights[i * len(prob_grid) + j, top]
            if not w.any():
                continue
            q = np.zeros((len(top), n), dtype=np.float32)
            previous = np.zeros_like(q)
            hypothesis = np.zeros_like(report)  # report probabilities under this (start, probability)
            for t in range(int(tau), t_obs):
                live = begin == t
                q[rows[live], top[live]] = 1.0
                hypothesis[:, :, t - ctx.t_lo] = q - previous
                previous = q.copy()
                matrix = contacts[t - ctx.t_lo]
                if matrix is not None:
                    q = 1.0 - (1.0 - q) * np.exp(np.log1p(-np.float32(p) * q) @ matrix)
            hypothesis[:, :, -1] = 1.0 - previous  # not infected before t_observe
            report += w[:, None, None] * hypothesis
            within += w @ _entropy(np.clip(hypothesis, 0.0, 1.0))
    return report[:, 0, :].sum(axis=1), report, within  # mass is the same whichever v is summed


def information_gain(ctx: Context, **kwargs) -> np.ndarray:
    """Mutual information between the source and each person's report (0 for people already tested)."""
    ob = ctx.outbreak
    found = _reports(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report, _ = found
    conditional = report / np.maximum(mass, 1e-300)[:, None, None]
    marginal = report.sum(axis=0) / mass.sum()
    gain = _entropy(marginal) - (mass @ _entropy(conditional)) / mass.sum()
    gain[ob.observers] = 0.0
    return np.maximum(gain, 0.0)


def joint_information_gain(ctx: Context, **kwargs) -> np.ndarray:
    """AKLD of Sterchi et al. (Sci Rep 2023): mean KL divergence between each full hypothesis's
    prediction of v's report and the consensus, sum_h w_h KL(P_h || P) = I(H; Y_v), with
    hypotheses H = (source, start time, probability). Unlike `information_gain`, it also
    rewards tests that only pin down the start time or rate. Same mean-field predictions,
    so only the choice rule differs. 0 for people already tested."""
    found = _reports(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report, within = found
    gain = _entropy(report.sum(axis=0) / mass.sum()) - within / mass.sum()
    gain[ctx.outbreak.observers] = 0.0
    return np.maximum(gain, 0.0)


def _reports_simulated(ctx: Context, n_runs: int = 10_000, max_sources: int = 100,
                       prob_grid: tuple[float, ...] = PROBS, n_start_times: int = 8, seed: int = 0):
    """Like `_reports`, but as Sterchi et al. (2023) do it: each hypothesis's report
    probabilities are frequencies over `n_runs` simulated outbreaks, and its weight is the
    product of those frequencies over the reports so far (their naive-Bayes posterior)."""
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    rng = np.random.default_rng(seed)
    y = ob.observed_times.astype(int)
    starts = _start_times(ctx, n_start_times)
    candidates = np.flatnonzero(np.isfinite(mean_field_loglik(ctx, prob_grid=prob_grid, n_start_times=n_start_times)).any(axis=0))
    n_times = t_obs - ctx.t_lo
    hypotheses, log_weight = [], []
    for tau in starts:
        for p in prob_grid:
            for s in candidates:
                begin = max(int(tau), int(graph.join[s]))
                if begin > y[y < t_obs].min():
                    continue
                start = np.full((n_runs, n), np.inf)
                start[:, s] = begin
                times = np.minimum(spread(graph, start, begin, t_obs, p, rng), t_obs)
                cls = np.where(times < t_obs, times - ctx.t_lo, n_times).astype(int)
                freq = np.zeros((n, n_times + 1))
                np.add.at(freq, (np.broadcast_to(np.arange(n), cls.shape), cls), 1.0 / n_runs)
                seen = freq[ob.observers, np.where(y < t_obs, y - ctx.t_lo, n_times)]
                hypotheses.append((s, freq))
                log_weight.append(np.log(np.maximum(seen, 1e-4)).sum())
    if not hypotheses:
        return None
    log_weight = np.array(log_weight)
    weight = np.exp(log_weight - logsumexp(log_weight))
    source_mass = {}
    for (s, _), w in zip(hypotheses, weight):
        source_mass[s] = source_mass.get(s, 0.0) + w
    top = sorted(source_mass, key=source_mass.get, reverse=True)[:max_sources]
    row = {s: i for i, s in enumerate(top)}
    report = np.zeros((len(top), n, n_times + 1))
    within = np.zeros(n)
    for (s, freq), w in zip(hypotheses, weight):
        if s in row:
            report[row[s]] += w * freq
            within += w * _entropy(freq)
    return report[:, 0, :].sum(axis=1), report, within


def simulated_joint_information_gain(ctx: Context, **kwargs) -> np.ndarray:
    """AKLD exactly as Sterchi et al. (2023): mean KL disagreement over (source, start, rate)
    hypotheses, with report probabilities from 10,000 simulations per hypothesis."""
    found = _reports_simulated(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report, within = found
    gain = _entropy(report.sum(axis=0) / mass.sum()) - within / mass.sum()
    gain[ctx.outbreak.observers] = 0.0
    return np.maximum(gain, 0.0)


def edge_cutting(ctx: Context, **kwargs) -> np.ndarray:
    """EC^2 gain (Golovin, Krause & Ray, NeurIPS 2010) of testing each person.

    Hypotheses are (source, outcome of every test), weighted by the current posterior
    w_s times the mean-field probability of the outcomes; regions are sources. Every
    pair of hypotheses with different sources is joined by an edge of weight equal to
    the product of their weights. Report c at person v removes every hypothesis that
    disagrees with it, so source s keeps weight w_s P(c | s) and the edge weight left is
        E_c = (m_c^2 - sum_s w_s^2 P(c | s)^2) / 2,   m_c = sum_s w_s P(c | s).
    The gain is E_before - sum_c m_c E_c. This objective is adaptive submodular, so
    testing greedily needs at most (2 ln(1 / p_min) + 1) times the expected number of
    tests of the best adaptive policy to pin down the source (under the mean-field
    outcome model, p_min = smallest hypothesis weight). Information gain has no such
    guarantee. 0 for people already tested."""
    found = _reports(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report, _ = found
    w = mass / mass.sum()
    conditional = report / np.maximum(mass, 1e-300)[:, None, None]  # P(c | s) per person
    m = np.einsum("s,svc->vc", w, conditional)
    squares = np.einsum("s,svc->vc", w ** 2, conditional ** 2)
    before = 0.5 * (1.0 - np.sum(w ** 2))
    gain = before - 0.5 * np.sum(m * (m ** 2 - squares), axis=1)
    gain[ctx.outbreak.observers] = 0.0
    return np.maximum(gain, 0.0)


def infection_probability(ctx: Context, **kwargs) -> np.ndarray:
    """Posterior probability that each person was infected before t_observe (0 for people
    already tested). Testing the most likely infected follows Baker et al. (PNAS 2021),
    who rank people for testing by their inferred infection probability."""
    found = _reports(ctx, **kwargs)
    if found is None:
        return np.zeros(ctx.graph.n_nodes)
    mass, report, _ = found
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
