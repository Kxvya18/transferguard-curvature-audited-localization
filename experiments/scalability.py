"""Cost of mean field on large temporal networks: time, peak memory and accuracy against size.

Networks (SNAP, open): sx-mathoverflow (24,818 users) and sx-askubuntu (159,316 users),
answer/comment interactions binned by day. Sub-networks of the N most active users
(N = 1,000 ... all) show how the cost grows. Outbreaks: SI at p = 0.3, 7-day horizon, 14-day
look-back, 1% of users observed (at least 10). Reported per outbreak: mean-field seconds,
peak memory (tracemalloc), candidates left after pruning, rank of the source, and the rank
given by distance center for reference.

Run:  python -m experiments.scalability <sx-mathoverflow|sx-askubuntu> <N or all> <count>
"""
import json
import sys
import time
import tracemalloc

import numpy as np

from experiments.benchmark_localization import PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_scores
from localization.spread import simulate_outbreak
from localization.temporal_graph import TemporalGraph, _join_times, load_snap_temporal

DAY = 86400
P, HORIZON, LOOKBACK, FRACTION = 0.3, 7, 14, 0.01


def top_users(graph, n):
    """Sub-network of the n users with the most contacts."""
    count = np.zeros(graph.n_nodes, dtype=np.int64)
    for w in graph.windows:
        if len(w):
            np.add.at(count, w.ravel(), 1)
    keep = np.sort(np.argsort(-count)[:n])
    index = np.full(graph.n_nodes, -1)
    index[keep] = np.arange(len(keep))
    windows = []
    for w in graph.windows:
        w = index[w] if len(w) else w
        windows.append(w[(w >= 0).all(axis=1)] if len(w) else w)
    return TemporalGraph(f"{graph.name} top {n}", len(keep), windows, _join_times(len(keep), windows))


def main(name, size, count):
    graph = load_snap_temporal(PROJECT_ROOT / "data" / "raw" / "temporal" / f"{name}.txt.gz", name, DAY)
    if size != "all":
        graph = top_users(graph, int(size))
    rng = np.random.default_rng(1201)
    out = PROJECT_ROOT / "outputs" / "scalability" / f"{name}_{size}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    while k < count:
        ob = simulate_outbreak(graph, rng, P, HORIZON, FRACTION, min_infected_observers=3, max_tries=2000)
        if ob is None:
            continue
        k += 1
        ctx = est.build_context(graph, ob, LOOKBACK)
        tracemalloc.start()
        start = time.time()
        try:
            score = mean_field_scores(ctx)
            error = None
        except MemoryError as e:
            score, error = None, repr(e)
        seconds = time.time() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        row = {"network": name, "users": graph.n_nodes, "contacts": int(sum(len(w) for w in graph.windows)),
               "observers": int(len(ob.observers)), "infected observers": int(np.sum(ob.observed_times < ob.t_observe)),
               "candidates": int(np.sum(score > est.INFEASIBLE)) if score is not None else None,
               "Mean-field seconds": seconds, "Mean-field peak MB": peak / 2**20, "error": error,
               "Mean-field": rank_of_source(score, ob.source) if score is not None else None,
               "Distance center": rank_of_source(est.distance_center(ctx), ob.source)}
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
