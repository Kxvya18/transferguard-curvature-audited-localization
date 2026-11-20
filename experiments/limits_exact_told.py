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
    ctx = replace(est.build_context(graph, ob, lookback), t_lo=ob.t_start)
    mf = mean_field_loglik(ctx, prob_grid=(p_true,), n_start_times=1)[0]
    cand = np.flatnonzero(np.isfinite(mf))
    n, y = graph.n_nodes, ob.observed_times
    joint = np.full(n, -np.inf); marg = np.full(n, -np.inf)
    for s in cand:
        start = np.full((M, n), np.inf); start[:, s] = max(ob.t_start, graph.join[s])
        times = np.minimum(spread(graph, start, ob.t_start, ob.t_observe, p_true, sim)[:, ob.observers], ob.t_observe)
        match = times == y[None, :]
        joint[s] = np.log(match.all(axis=1).mean() + 1e-12)
        marg[s] = np.log(match.mean(axis=0) + 1e-4).sum()
    row = {"network": name, "k": k, "M": M, "candidates": len(cand),
           "MF, told": rank_of_source(fin(mf), ob.source),
           "Exact marginals, told": rank_of_source(fin(marg), ob.source),
           "Exact joint, told": rank_of_source(fin(joint), ob.source),
           "joint all zero": bool(np.all(joint[cand] < -27))}
    with open(out, "a") as h: h.write(json.dumps(row) + "\n")
