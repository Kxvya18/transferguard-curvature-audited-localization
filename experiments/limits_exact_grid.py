"""How good can any localizer be? Exact likelihood by Monte Carlo on the same outbreaks.

Told the true start and rate:
- exact joint: P(every observer report matches exactly | s), by counting matching runs;
- exact marginals: each observer's exact report probability, multiplied (independence),
  which separates MF's marginal error from its independence assumption;
- MF told, for reference.
"""
import json, sys
from dataclasses import replace
import numpy as np
from experiments.benchmark_localization import DATASETS, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_loglik
from scipy.special import logsumexp
from localization.spread import simulate_outbreak, spread
name, count, M, shard, shards, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
loader, p_true, horizon, lookback = DATASETS[name]
graph = loader(901); rng = np.random.default_rng(901); sim = np.random.default_rng([7, shard])
fin = lambda s: np.where(np.isfinite(s), s, est.INFEASIBLE)
k = 0
while k < count:
    ob = simulate_outbreak(graph, rng, p_true, horizon, 0.1, max_tries=2000)
    if ob is None: continue
    k += 1
    if k % shards != shard: continue
    ctx = est.build_context(graph, ob, lookback)
    grid = mean_field_loglik(ctx)
    mf = logsumexp(grid, axis=0)
    cand = np.flatnonzero(np.isfinite(mf))
    n, y = graph.n_nodes, ob.observed_times
    first = int(y[y < ob.t_observe].min())
    starts = np.unique(np.linspace(ctx.t_lo, max(first, ctx.t_lo), 8).round().astype(int))
    joint = np.full((len(starts) * 5, n), -np.inf)
    for s in cand:
        g = 0
        for tau in starts:
            for p in (0.1, 0.3, 0.5, 0.7, 0.9):
                b = max(tau, graph.join[s])
                if b <= first:
                    start = np.full((M, n), np.inf); start[:, s] = b
                    times = np.minimum(spread(graph, start, b, ob.t_observe, p, sim)[:, ob.observers], ob.t_observe)
                    joint[g, s] = np.log((times == y[None, :]).all(axis=1).mean() + 1e-12)
                g += 1
    ex = logsumexp(joint, axis=0)
    row = {"network": name, "k": k, "M": M, "candidates": len(cand),
           "Mean-field": rank_of_source(fin(mf), ob.source),
           "Exact joint": rank_of_source(fin(ex), ob.source),
           "joint all zero": bool(np.all(ex[cand] < -27))}
    with open(out, "a") as h: h.write(json.dumps(row) + "\n")
