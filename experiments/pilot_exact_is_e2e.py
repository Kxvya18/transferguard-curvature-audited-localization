"""Pilot #1 stage 2 (docs/pilot_new_components.md): particle-filter likelihood end to end, start and
rate unknown, summed over the same grid as mean field. Outbreaks: the first 40 of the BP
comparison (seed 101, true rate, 10% observers).

Run:  python -m experiments.pilot_exact_is_e2e <network> [count] [particles]
"""
import json
import sys
import time

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization import exact_is
from localization.mean_field import _start_times, mean_field_scores
from localization.spread import simulate_outbreak

PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)  # mean field's default grid


def main(name, count=40, particles=200):
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(101)
    rng = np.random.default_rng(101)
    sim = np.random.default_rng(1501)
    folder = PROJECT_ROOT / "outputs" / "pilot_exact_is_e2e"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{name.replace(' ', '_')}.jsonl").open("w") as h:
        k = 0
        while k < count:
            ob = simulate_outbreak(graph, rng, p, horizon, 0.1)
            if ob is None:
                continue
            k += 1
            ctx = est.build_context(graph, ob, lookback)
            start = time.time()
            mf = mean_field_scores(ctx)
            mf_seconds = time.time() - start
            candidates = np.flatnonzero(mf > est.INFEASIBLE)
            starts = _start_times(ctx, 8)
            counts = exact_is.contact_counts(graph, int(starts.min()), ob.t_observe)
            exact = np.full(graph.n_nodes, est.INFEASIBLE)
            start = time.time()
            for s in candidates:
                terms = [exact_is.log_likelihood(graph, ob, s, tau, q, particles, sim, counts)[0]
                         for tau in starts for q in PROBS]
                total = logsumexp(terms)
                exact[s] = total if np.isfinite(total) else est.INFEASIBLE
            row = {"network": name, "outbreak": k - 1, "particles": particles, "candidates": int(len(candidates)),
                   "Mean-field": rank_of_source(mf, ob.source),
                   "Exact (particle filter)": rank_of_source(exact, ob.source),
                   "MF seconds": mf_seconds, "Exact seconds": time.time() - start}
            h.write(json.dumps(row) + "\n")
            h.flush()


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:4]))
