"""Confidence sets for the source by hypothesis testing (Dawkins, Li and Xu, ICML 2021), against
our split-conformal sets, on the outbreaks of the BP comparison.

Dawkins et al. test, for every candidate s, "s is the source" with a Monte Carlo p-value and
keep every candidate that is not rejected. The set covers the source with probability at least
1 - alpha for any test statistic and any network, with no calibration outbreaks. Here the
statistic is the mean-field log-likelihood of the reports at s, maximised over the start grid
(larger = more consistent). Under the null, M outbreaks are simulated from s with the TRUE
transmission rate (Dawkins et al. assume the spreading model is known; this favours them) and
the start drawn from the same start grid, observed at the same observers, and conditioned on at
least 3 infected observers, as the real outbreaks are. p(s) = (1 + #{sim stat <= obs stat}) / (M + 1).
Candidates are those mean field does not prune (Lemma 2: a pruned node has likelihood 0).

Run:  python -m experiments.confidence_sets <network> <count> [M] [shard shards]
"""
import json
import sys
import time
from dataclasses import replace

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import _start_times, mean_field_loglik, mean_field_scores
from localization.spread import Outbreak, simulate_outbreak, spread

ALPHA = 0.1


def statistic(ctx, s, p):
    """Mean-field log-likelihood of the reports at candidate s (rate p), best start."""
    return float(np.max(mean_field_loglik(ctx, prob_grid=(p,), candidates=np.array([s]))[:, s]))


def main(name, count, M=100, shard=0, shards=1):
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(101)
    rng = np.random.default_rng(101)  # same outbreaks as outputs/bp_comparison/<name>_0.1.jsonl (true rate)
    sim = np.random.default_rng([5, shard])
    out = PROJECT_ROOT / "outputs" / "confidence_sets" / f"{name.replace(' ', '_')}_part{shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    while k < count:
        ob = simulate_outbreak(graph, rng, p, horizon, 0.1)
        if ob is None:
            continue
        k += 1
        if (k - 1) % shards != shard:
            continue
        start = time.time()
        ctx = est.build_context(graph, ob, lookback)
        mf = mean_field_scores(ctx)
        candidates = np.flatnonzero(mf > est.INFEASIBLE)
        starts = _start_times(ctx, 8)
        first = int(ob.observed_times[ob.observed_times < ob.t_observe].min())
        kept = []
        for s in candidates:
            observed = statistic(ctx, s, p)
            null = []
            tries = 0
            while len(null) < M and tries < 20 * M:
                tries += 1
                tau = max(int(sim.choice(starts)), int(graph.join[s]))
                begin = np.full((1, graph.n_nodes), np.inf)
                begin[0, s] = tau
                times = np.minimum(spread(graph, begin, tau, ob.t_observe, p, sim)[0], ob.t_observe)
                y = times[ob.observers]
                if np.sum(y < ob.t_observe) < 3:
                    continue
                fake = replace(ctx, outbreak=Outbreak(s, tau, ob.t_observe, ob.observers, y))
                null.append(statistic(fake, s, p))
            pvalue = (1 + np.sum(np.array(null) <= observed)) / (len(null) + 1) if null else 1.0
            if pvalue > ALPHA:
                kept.append(int(s))
        row = {"network": name, "outbreak": k - 1, "M": M, "candidates": int(len(candidates)),
               "Dawkins set size": len(kept), "Dawkins covers": ob.source in kept,
               "Mean-field rank": rank_of_source(mf, ob.source), "seconds": time.time() - start}
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(a) for a in sys.argv[3:6]))
