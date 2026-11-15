"""Time-respecting arrival features for every candidate source.

For a candidate s and a possible start time tau, the earliest arrival of an
infection from s at each observer is computed along contacts in time order
(see `spread.earliest_arrival`). These arrivals are compared with what the
observers reported, and the best start time per candidate is kept. The result
is a fixed-length description of how well each node explains the observations
physically, which a learned model can then weigh.
"""

from __future__ import annotations

import numpy as np

from .estimators import Context
from .spread import earliest_arrival

FEATURE_NAMES = [
    "feasible",  # joined no later than the first observed infection
    "fit_residual",  # RMS residual of observed times vs start + speed * delay
    "fit_speed",  # fitted windows per unit of earliest-arrival delay
    "start_lead",  # fitted start, as a fraction of the lookback before t_observe
    "reach_infected",  # fraction of infected observers reachable in time
    "reach_healthy_early",  # fraction of healthy observers it could have reached with time to spare
    "min_delay",  # shortest arrival delay to an infected observer (lookback units)
    "mean_delay",  # mean arrival delay to infected observers (lookback units)
]


def _row_fits(delays: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares fit times ≈ a + b * delay for every row at once (b >= 0).

    Rows with any infinite delay get an infinite residual.
    """
    ok = np.all(np.isfinite(delays), axis=1)
    x = np.where(np.isfinite(delays), delays, 0.0)
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = times.mean()
    var = ((x - x_mean) ** 2).mean(axis=1)
    cov = ((x - x_mean) * (times - y_mean)).mean(axis=1)
    b = np.where(var > 1e-12, np.maximum(cov / np.maximum(var, 1e-12), 0.0), 0.0)
    a = y_mean - b * x_mean[:, 0]
    rss = ((times - (a[:, None] + b[:, None] * x)) ** 2).mean(axis=1)
    return np.where(ok, np.sqrt(rss), np.inf), a, b


def arrival_features(ctx: Context, n_start_times: int = 8) -> np.ndarray:
    """Feature matrix of shape (n_nodes, len(FEATURE_NAMES))."""
    ob = ctx.outbreak
    graph = ctx.graph
    lookback = max(ob.t_observe - ctx.t_lo, 1)
    times = ctx.infected_times
    infected = ctx.infected
    healthy = ob.observers[ob.observed_times >= ob.t_observe]
    first = int(times.min())
    starts = np.unique(np.linspace(ctx.t_lo, max(first, ctx.t_lo), n_start_times).round().astype(int))

    n = graph.n_nodes
    best = np.full(n, np.inf)
    feats = np.zeros((n, len(FEATURE_NAMES)))
    for tau in starts:
        arrival = earliest_arrival(graph, int(tau), ob.t_observe)
        begin = np.maximum(tau, graph.join).astype(float)
        delays = arrival[:, infected] - begin[:, None]
        resid, a, b = _row_fits(delays, times)
        reach_inf = np.isfinite(delays).mean(axis=1)
        if len(healthy):
            early = arrival[:, healthy] < ob.t_observe - 1
            reach_healthy = early.mean(axis=1)
        else:
            reach_healthy = np.zeros(n)
        finite = np.where(np.isfinite(delays), delays, np.nan)
        with np.errstate(all="ignore"):
            min_d = np.nanmin(finite, axis=1)
            mean_d = np.nanmean(finite, axis=1)
        # Keep each node's best start time; unreachable nodes keep their reach stats.
        better = (resid < best) | (~np.isfinite(best) & (reach_inf > feats[:, 4]))
        best = np.where(resid < best, resid, best)
        feats[better, 1] = np.where(np.isfinite(resid[better]), resid[better] / lookback, 2.0)
        feats[better, 2] = b[better]
        feats[better, 3] = np.clip((ob.t_observe - a[better]) / lookback, -1.0, 2.0)
        feats[better, 4] = reach_inf[better]
        feats[better, 5] = reach_healthy[better]
        feats[better, 6] = np.nan_to_num(min_d[better] / lookback, nan=2.0)
        feats[better, 7] = np.nan_to_num(mean_d[better] / lookback, nan=2.0)
    feats[:, 0] = (graph.join <= first).astype(float)
    feats[~np.isfinite(best), 1] = 2.0
    return feats.astype(np.float32)
