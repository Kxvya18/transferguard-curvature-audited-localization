"""MF vs BP on the same outbreaks as mf_variants.py (same seed), sharded."""
import json, sys, time
import numpy as np
from experiments.benchmark_localization import DATASETS, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_scores
from localization.belief_propagation import bp_scores
from localization.spread import simulate_outbreak
name, count, shard, shards, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
loader, p_true, horizon, lookback = DATASETS[name]
graph = loader(901); rng = np.random.default_rng(901)
k = 0
while k < count:
    ob = simulate_outbreak(graph, rng, p_true, horizon, 0.1, max_tries=2000)
    if ob is None: continue
    k += 1
    if k % shards != shard: continue
    ctx = est.build_context(graph, ob, lookback)
    row = {"network": name, "k": k}
    for label, fn in (("Mean-field", mean_field_scores), ("Belief propagation", bp_scores)):
        t = time.time(); row[label] = rank_of_source(fn(ctx), ob.source); row[label + " seconds"] = time.time() - t
    with open(out, "a") as h: h.write(json.dumps(row) + "\n")
