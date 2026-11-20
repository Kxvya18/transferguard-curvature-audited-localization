"""Where does MF lose accuracy? MF-only variants on many outbreaks (cheap, no BP)."""
import json, sys, time
from dataclasses import replace
import numpy as np
from scipy.special import logsumexp
from experiments.benchmark_localization import DATASETS, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_loglik, filtered_loglik, message_passing_scores
from localization.spread import simulate_outbreak

name, count, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
loader, p_true, horizon, lookback = DATASETS[name]
seed = 901
graph = loader(seed); rng = np.random.default_rng(seed)
fin = lambda s: np.where(np.isfinite(s), s, est.INFEASIBLE)
P5 = (0.1, 0.3, 0.5, 0.7, 0.9)
P9 = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9)
k = 0
while k < count:
    ob = simulate_outbreak(graph, rng, p_true, horizon, 0.1, max_tries=2000)
    if ob is None: continue
    k += 1
    ctx = est.build_context(graph, ob, lookback)
    told = replace(ctx, t_lo=ob.t_start)
    row = {"network": name, "n": graph.n_nodes}
    t = time.time()
    base = mean_field_loglik(ctx)
    row["MF (8 starts x 5 rates)"] = rank_of_source(fin(logsumexp(base, 0)), ob.source)
    row["seconds"] = time.time() - t
    row["MF, best (start,rate) only"] = rank_of_source(fin(base.max(0)), ob.source)
    row["MF, all starts x 9 rates"] = rank_of_source(fin(logsumexp(mean_field_loglik(ctx, prob_grid=P9, n_start_times=None), 0)), ob.source)
    for fl in (1e-8, 1e-2):
        row[f"MF, floor {fl:g}"] = rank_of_source(fin(logsumexp(mean_field_loglik(ctx, floor=fl), 0)), ob.source)
    row["MF, true rate only"] = rank_of_source(fin(logsumexp(mean_field_loglik(ctx, prob_grid=(p_true,)), 0)), ob.source)
    row["MF, true start only"] = rank_of_source(fin(logsumexp(mean_field_loglik(told, n_start_times=1), 0)), ob.source)
    row["MF, told both"] = rank_of_source(fin(mean_field_loglik(told, prob_grid=(p_true,), n_start_times=1)[0]), ob.source)
    row["Filtered MF"] = rank_of_source(fin(logsumexp(filtered_loglik(ctx), 0)), ob.source)
    row["Message passing"] = rank_of_source(message_passing_scores(ctx), ob.source)
    row["candidates"] = int(np.sum(np.isfinite(logsumexp(base, 0))))
    with open(out, "a") as h: h.write(json.dumps(row) + "\n")
