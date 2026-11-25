"""Is belief propagation set up fairly? Damping and convergence on the stored outbreaks.

Regenerates the first COUNT true-rate outbreaks of the BP comparison (seed 101, so the same
outbreaks as outputs/bp_comparison/<network>_0.1.jsonl) and scores each with BP at several
damping values, recording the source's rank and the share of (start, rate) runs in which BP
did not converge (final message change above the tolerance).

Run:  python -m experiments.bp_sensitivity <network> <count> <damping> [shard shards]
"""
import json
import sys
import time

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.belief_propagation import bp_scores
from localization.spread import simulate_outbreak


def main(name, count, damping, shard=0, shards=1):
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(101)
    rng = np.random.default_rng(101)
    out = PROJECT_ROOT / "outputs" / "bp_sensitivity" / f"{name.replace(' ', '_')}_damping{damping}_part{shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    while k < count:
        ob = simulate_outbreak(graph, rng, p, horizon, 0.1)  # same draws as compare_bp (true rate first)
        if ob is None:
            continue
        k += 1
        if (k - 1) % shards != shard:
            continue
        ctx = est.build_context(graph, ob, lookback)
        start = time.time()
        loglik, log_seed, errors = bp_scores(ctx, damping=damping, parts=True)
        score = logsumexp(loglik[:, None] + log_seed, axis=0)
        score = np.where(graph.join <= int(ctx.infected_times.min()), score, -np.inf)
        score = np.where(np.isfinite(score), score, est.INFEASIBLE)
        row = {"network": name, "outbreak": k - 1, "damping": damping, "rank": rank_of_source(score, ob.source),
               "share not converged": float(np.mean(~(np.asarray(errors) <= 1e-6))), "seconds": time.time() - start}
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), *(int(a) for a in sys.argv[4:6]))
