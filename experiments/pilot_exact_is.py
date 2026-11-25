"""Pilot #1 (docs/pilot_new_components.md): exact likelihood by a guided particle filter, told the
true start and rate, on the outbreaks of `limits_exact_told.py` (seed 901).

Run:  python -m experiments.pilot_exact_is <network> [count] [particles]
"""
import json
import sys
import time
from dataclasses import replace

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization import exact_is
from localization.mean_field import mean_field_loglik
from localization.spread import simulate_outbreak


def main(name, count=40, particles=1000):
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(901)
    rng = np.random.default_rng(901)
    sim = np.random.default_rng(1401)
    folder = PROJECT_ROOT / "outputs" / "pilot_exact_is"
    folder.mkdir(parents=True, exist_ok=True)
    fin = lambda s: np.where(np.isfinite(s), s, est.INFEASIBLE)
    with (folder / f"{name.replace(' ', '_')}.jsonl").open("w") as h:
        k = 0
        while k < count:
            ob = simulate_outbreak(graph, rng, p, horizon, 0.1, max_tries=2000)
            if ob is None:
                continue
            k += 1
            ctx = replace(est.build_context(graph, ob, lookback), t_lo=ob.t_start)
            start = time.time()
            mf = mean_field_loglik(ctx, prob_grid=(p,), n_start_times=1)[0]
            mf_seconds = time.time() - start
            candidates = np.flatnonzero(np.isfinite(mf))
            counts = exact_is.contact_counts(graph, ob.t_start, ob.t_observe)
            exact = np.full(graph.n_nodes, -np.inf)
            ess = []
            start = time.time()
            for s in candidates:
                exact[s], share = exact_is.log_likelihood(graph, ob, s, ob.t_start, p, particles, sim, counts)
                ess.append(share)
            row = {"network": name, "outbreak": k - 1, "particles": particles, "candidates": int(len(candidates)),
                   "MF, told": rank_of_source(fin(mf), ob.source),
                   "Exact (particle filter), told": rank_of_source(fin(exact), ob.source),
                   "source finite": bool(np.isfinite(exact[ob.source])),
                   "MF seconds": mf_seconds, "Exact seconds": time.time() - start,
                   "median final ESS share": float(np.median(ess)) if ess else None}
            h.write(json.dumps(row) + "\n")
            h.flush()


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:4]))
