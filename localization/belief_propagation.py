"""Belief-propagation baseline (Braunstein & Ingrosso, Sci. Rep. 2016) via the SIB library.

SIB runs belief propagation over infection-time trajectories on a list of timed
contacts; it is exact on tree-like contact patterns and conditions on every
observation, forwards and backwards. It needs the `sib` package, built separately
(see README); the rest of this package does not depend on it.

Time convention (checked against simulation): a contact in our window t is SIB
contact time t + 1, and a node infected at our time y has SIB infection time y. A SIB
"seed" (infection time -1) is a source that is infected when the outbreak starts.
For each start window tau and transmission probability p on the same grids as the
mean-field localizer, SIB returns the log-likelihood of the observations and each
node's posterior probability of being a seed; these are combined as
    score(s) = log sum over (tau, p) of exp(loglik) * P(s is a seed | tau, p).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .estimators import INFEASIBLE, Context
from .mean_field import _start_times


def bp_scores(
    ctx: Context,
    prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    n_start_times: int | None = 8,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    damping: float = 0.5,
    status_only: bool = False,
    parts: bool = False,
):
    """With `parts`, returns per (start, rate) term: the Bethe log-likelihood, the log seed
    marginals and BP's final convergence error, instead of the combined score.

    Damping 0.5 is needed on dense, loopy contact networks (undamped BP oscillates)."""
    import sib

    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs = graph.n_nodes, ob.t_observe
    y = ob.observed_times.astype(int)
    terms, logliks, seeds, errors = [], [], [], []
    for tau in _start_times(ctx, n_start_times):
        for p in prob_grid:
            # SIB needs each node's events in time order, so contacts and tests are
            # appended together, sorted by time.
            events = []
            for t in range(int(tau), t_obs):
                for i, j in graph.windows[t]:
                    events.append((t + 1, 1, int(i), int(j)))
            if not events:
                continue
            # In SIB a node with infection time ti is S at times <= ti and I after, so
            # "infected at y" is S at y and I at y + 1; a source (seed, ti = -1) is
            # only constrained by the I test. "Healthy" (not infected before t_obs) is S at t_obs.
            for node, time in zip(ob.observers, y):
                if time < t_obs:
                    events.append((int(time) + 1, 0, int(node), 1))
                    if time > tau and not status_only:  # status-only: infected by t_obs, time unknown
                        events.append((int(time), 0, int(node), 0))
                else:
                    events.append((t_obs, 0, int(node), 0))
            params = sib.Params(prob_i=sib.Uniform(1.0), prob_r=sib.Exponential(1e-10), pseed=1.0 / n, psus=0.5)
            factor = sib.FactorGraph(params=params)
            for time, kind, a, b in sorted(events):
                if kind == 1:
                    factor.append_contact(a, b, time, float(p), float(p))
                else:
                    factor.append_observation(a, sib.Test(1, 0, 0) if b == 0 else sib.Test(0, 1, 0), time)
            last = [np.nan]  # final change in messages; above `tolerance` means BP did not converge
            factor.iterate(maxit=max_iterations, tol=tolerance, damping=damping,
                           callback=lambda t, err, f: last.__setitem__(0, err))
            error = last[0]
            seed = np.zeros(n)
            for k in range(min(len(factor.nodes), n)):
                seed[k] = factor.nodes[k].bt[0]
            with np.errstate(divide="ignore"):
                terms.append(factor.loglikelihood() + np.log(seed))
                logliks.append(factor.loglikelihood())
                seeds.append(np.log(seed))
            errors.append(float(error) if error is not None else np.nan)
    if parts:
        return np.array(logliks), np.array(seeds), np.array(errors)
    score = logsumexp(np.array(terms), axis=0) if terms else np.full(n, -np.inf)
    score = np.where(graph.join <= int(ctx.infected_times.min()), score, -np.inf)
    return np.where(np.isfinite(score), score, INFEASIBLE)


def hybrid_scores(ctx: Context, prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
                  n_start_times: int | None = 8, **kwargs) -> np.ndarray:
    """Mean field chooses the model, belief propagation finds the source.

    The start time and transmission probability are unknown, so a localizer must
    weigh each (start, rate) pair by how well it explains the reports. BP's Bethe
    estimate of that evidence is unreliable on loopy contact networks, where BP often
    does not converge for the wrong (start, rate); mean field is a forward recursion
    that always gives a finite evidence. So each pair g gets the mean-field weight
    P(g | reports) = sum_s exp(loglik_MF[g, s]) / Z, and within it BP's posterior
    probability that s is the seed:
        score(s) = log sum_g P(g | reports) * P_BP(s is the seed | g, reports).
    """
    from .mean_field import mean_field_loglik

    mf = mean_field_loglik(ctx, prob_grid=prob_grid, n_start_times=n_start_times)
    _, log_seed, _ = bp_scores(ctx, prob_grid=prob_grid, n_start_times=n_start_times, parts=True, **kwargs)
    if len(log_seed) != len(mf):  # a (start, rate) pair without contacts was skipped by BP
        return mean_field_scores_fallback(mf)
    evidence = logsumexp(mf, axis=1)
    weight = evidence - logsumexp(evidence[np.isfinite(evidence)]) if np.isfinite(evidence).any() else evidence
    score = logsumexp(weight[:, None] + log_seed, axis=0)
    score = np.where(ctx.graph.join <= int(ctx.infected_times.min()), score, -np.inf)
    return np.where(np.isfinite(score), score, INFEASIBLE)


def mean_field_scores_fallback(mf: np.ndarray) -> np.ndarray:
    score = logsumexp(mf, axis=0)
    return np.where(np.isfinite(score), score, INFEASIBLE)
